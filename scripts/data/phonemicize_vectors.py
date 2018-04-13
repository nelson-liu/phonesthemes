from argparse import ArgumentParser
from collections import OrderedDict
import logging
import mmap
import operator
import os

import numpy as np
from tqdm import tqdm


def main(vectors_path, graphemes_to_phonemes_path, output_dir):
    # Get a dictionary mapping string graphemes to tuple phonemes, as well as
    # a dictionary mapping phoneme representation to their GloVe vectors.
    graphemes_to_phonemes, phonemicized_vectors = phonemicize_vectors(
        vectors_path, graphemes_to_phonemes_path)

    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        logger.info("output dir {} does not exist, creating it".format(output_dir))
        os.makedirs(output_dir)

    vector_filename = os.path.splitext(os.path.basename(vectors_path))[0]
    phonemicized_vectors_path = os.path.join(
        output_dir, vector_filename + ".phonemicized.txt")
    logger.info("Writing phonemicized vectors to {}".format(phonemicized_vectors_path))
    with open(phonemicized_vectors_path, "w") as phonemicized_vectors_file:
        # Convert the dictionary to a list of tuples for reproducibility
        sorted_phonemicized_vectors = sorted(
            phonemicized_vectors.items(), key=operator.itemgetter(0))
        for phonemicized_word, vector in sorted_phonemicized_vectors:
            phonemicized_vectors_file.write("{} {}\n".format(
                ",".join(phonemicized_word),
                " ".join([str(x) for x in vector])))

    vector_g2p_path = os.path.join(
        output_dir, vector_filename + ".graphemes_to_phonemes.txt")
    logger.info("Writing grapheme to phoneme mappings used for "
                "phonemicization to {}.".format(vector_g2p_path))
    with open(vector_g2p_path, "w") as vector_g2p_file:
        # Convert the dictionary to a list of tuples for reproducibility
        sorted_graphemes_to_phonemes = sorted(
            graphemes_to_phonemes.items(), key=operator.itemgetter(0))
        for grapheme, phonemes in sorted_graphemes_to_phonemes:
            vector_g2p_file.write("{}\t{}\n".format(grapheme, " ".join(phonemes)))


def phonemicize_vectors(vectors_path, graphemes_to_phonemes_path):
    # Read the vectors from vectors_path
    logger.info("Reading vectors from {}".format(vectors_path))
    with open(vectors_path) as vectors_file:
        # Dictionary of word to vector
        vectors = OrderedDict()
        for line in tqdm(vectors_file,
                         total=get_line_number(vectors_path)):
            split_line = line.rstrip("\n").split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            vectors[word] = embedding
    logger.info("Loaded {} vectors.".format(len(vectors)))

    # Load the graphemes to phonemes data.
    logger.info((
        "Loading graphemes to phonemes data at {} and mapping them to "
        "the associated GloVe vector. Preserving only words in the "
        "input glove vectors.").format(graphemes_to_phonemes_path))
    with open(graphemes_to_phonemes_path) as graphemes_to_phonemes_file:
        # graphemes_to_phonemes is a dictionary of {grapheme string:
        # phoneme representation tuple}
        graphemes_to_phonemes = {}

        # phonemicized_vectors is a dictionary of
        # {phoneme representation tuple: GloVe vector}
        phonemicized_vectors = {}

        # vector_vocabulary is a set of all of the words in the vectors.
        vector_vocabulary = set(vectors.keys())

        # homophones is the set of all phoneme representation
        # tuples that occur multiple times in the vectors.
        homophones = set()

        # The total number of homophones discarded.
        num_homophones_discarded = 0

        for line in tqdm(graphemes_to_phonemes_file,
                         total=get_line_number(graphemes_to_phonemes_path)):
            split_line = line.rstrip("\n").split()
            # The first element of the split line is the original word:
            word = split_line[0]
            # The rest of the line is the phoneme representation.
            # We turn it into a tuple.
            phoneme_representation = tuple([str(phone) for phone in split_line[1:]])

            if word in vector_vocabulary:
                graphemes_to_phonemes[word] = phoneme_representation
                # Homophones mean that one phoneme representation can have
                # multiple vectors. Thus, we discard homophones.
                if phoneme_representation in homophones:
                    num_homophones_discarded += 1
                else:
                    if phoneme_representation in phonemicized_vectors:
                        homophones.add(phoneme_representation)
                        del phonemicized_vectors[phoneme_representation]
                        num_homophones_discarded += 2
                        continue
                    phonemicized_vectors[phoneme_representation] = vectors[word]

    logger.info("Loaded {} grapheme to phoneme entries.".format(
        len(graphemes_to_phonemes)))
    logger.info("Discarded {} homophones, ({} different words)".format(
        len(homophones), num_homophones_discarded))
    logger.info("Created {} phonemecized vectors".format(len(phonemicized_vectors)))
    return (graphemes_to_phonemes, phonemicized_vectors)


def get_line_number(file_path):
    """Fast function to calculate total number of lines in text file."""
    # from http://stackoverflow.com/a/850962/1877942
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    script_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    project_root_dir = os.path.abspath(os.path.join(script_dir,
                                                    os.pardir, os.pardir))

    parser = ArgumentParser(description=(
        "Given a file of vectors (string and dimensions, space separated), "
        "generate a file of phonemes to vectors, where the phonemes are "
        "comma-separated. In addition, generate a file of string to phonemes"
        "where the keys are the graphemes and the values are the "
        "phoneme representations."))
    parser.add_argument("--vectors_path",
                        metavar="<vectors_path>", type=str,
                        help=("Path to a file of vectors to phonemicize."))
    parser.add_argument("--graphemes_to_phonemes_path",
                        metavar="<graphemes_to_phonemes_path>", type=str,
                        help=("Path to a file with words and "
                              "their phonemicized representations"))
    parser.add_argument("--output_dir",
                        metavar="<output_dir>", type=str,
                        help=("Folder to output the file of "
                              "<comma separated phonemes><space>"
                              "<space-delimited numpy array> and the file "
                              "of <word><space><comma separated phonemes>."))

    A = parser.parse_args()
    main(A.vectors_path, A.graphemes_to_phonemes_path,
         A.output_dir)
