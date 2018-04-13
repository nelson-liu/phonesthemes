from argparse import ArgumentParser
from collections import Counter
import logging
import os
import operator
import re
import subprocess


def get_morph_segmentations(celex_morph_data_path, column):
    """
    Given the path to the CELEX2 morphology and lemmas data
    for a language, extract a dictionary of word to list of
    constituent morphemes.
    """
    # Find the .cd file with the data we wish to extract
    logger.info("Searching in {} for a data file (file "
                "with .cd extension)".format(celex_morph_data_path))
    data_path = [os.path.join(celex_morph_data_path, path) for path in
                 os.listdir(celex_morph_data_path) if path.endswith(".cd")]
    if len(data_path) != 1:
        raise ValueError("Found possible data files {}. Ensure that there "
                         "is only one file with the .cd extension in the "
                         "directory.")
    data_path = data_path[0]
    # Get the path to the stripstr.awk script
    stripstr_path = os.path.join(celex_morph_data_path, "awk", "stripstr.awk")
    # Build the command to run.
    command = ["awk", "-f", stripstr_path, data_path, str(column)]
    logger.info("Running command {}".format(command))
    raw_morph_segmentations = subprocess.check_output(
        command, universal_newlines=True)
    # Split raw morph segmentations by newline, and iterate over the raw
    # morph segmentations + the original morpheme data to get a mapping from
    # words to segmentations.
    morph_segmentations = {}
    with open(data_path) as data_file:
        for data_line, segmentation in zip(
                data_file, raw_morph_segmentations.split("\n")):
            word = data_line.split("\\")[1]
            if not len(segmentation.strip()) == 0:
                # Split segmentation on "+" and strip the resultant morphemes
                segmentation_morphemes = [morpheme.strip() for morpheme in
                                          re.split(r"\+| ", segmentation)]
                morph_segmentations[word] = segmentation_morphemes
    return morph_segmentations


def main(celex_morph_data_path, column, output_dir):
    # Extract the morphemes from the celex data by running the
    # "stripstr.awk" script in the "awk" subdirectory.
    morph_segmentations = get_morph_segmentations(celex_morph_data_path, column)

    morpheme_frequencies = Counter()
    for morphemes in morph_segmentations.values():
        for morpheme in morphemes:
            morpheme_frequencies[morpheme] += 1

    # Bound morphemes are those that occur at least 40 times
    bound_morphemes = [morph for morph, count in
                       sorted(morpheme_frequencies.items(), reverse=True,
                              key=operator.itemgetter(1, 0)) if
                       count >= 40]
    celex_data_name = os.path.basename(os.path.normpath(
        celex_morph_data_path))

    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        logger.info("output dir {} does not exist, creating it".format(output_dir))
        os.makedirs(output_dir)

    morph_segmentations_path = os.path.join(
        output_dir, celex_data_name + ".seg")
    logger.info("Saving segmentation dictionary to {}".format(
        morph_segmentations_path))
    with open(morph_segmentations_path, "w") as morph_segmentations_file:
        # Convert dictionary to sorted list of tuples for reproducibility
        sorted_segmentations = sorted(
            morph_segmentations.items(), key=operator.itemgetter(0))
        for word, segmentation in sorted_segmentations:
            morph_segmentations_file.write("{}\t{}\n".format(
                word, " ".join(segmentation)))

    bound_morphemes_path = os.path.join(
        output_dir, celex_data_name + ".bound_morphemes.txt")
    logger.info("Writing list of bound morphemes to {}".format(
        bound_morphemes_path))
    with open(bound_morphemes_path, "w") as bound_morphemes_file:
        for bound_morpheme in bound_morphemes:
            bound_morphemes_file.write("{}\n".format(bound_morpheme))
    logger.info("Saved list of bound morphemes to {}, "
                "{} items in total".format(
                    bound_morphemes_path, len(bound_morphemes)))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    script_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    project_root_dir = os.path.abspath(os.path.join(script_dir,
                                                    os.pardir, os.pardir))

    parser = ArgumentParser(description=("Extract a list of bound morphemes from "
                                         "the morphology and lemmas data in CELEX2."))
    parser.add_argument("--celex_morph_data_path",
                        metavar="<celex_morph_data_path>", type=str, required=True,
                        help=("Path to CELEX2 language morphology and lemmas data. "
                              "This folder is generally named *ml, e.g. eml/ for "
                              "english, gml/ for german, and dml/ for dutch"))
    parser.add_argument("--column", metavar="<column>", type=int, required=True,
                        help=("The column in the data with the morpheme segmentation to "
                              "flatten. For English CELEX, this is 22. For German, it "
                              "is 14. For Dutch, it is 13."))
    parser.add_argument("--output_dir",
                        metavar="<output_dir>", type=str, required=True,
                        help=("Folder to output the pickled dictionary of "
                              "{word:numpy array of vector} and a list of "
                              "words in the dictionary that were kept."))
    A = parser.parse_args()
    main(A.celex_morph_data_path, A.column, A.output_dir)
