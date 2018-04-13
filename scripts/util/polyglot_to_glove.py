import io
import logging
import os
import pickle

from argparse import ArgumentParser
import six


def main(polyglot_path, output_dir):
    # Read the polyglot vectors
    if six.PY2:
        words, vectors = pickle.load(open(polyglot_path, "rb"))
    else:
        words, vectors = pickle.load(open(polyglot_path, "rb"), encoding='latin1')

    polyglot_filename = os.path.splitext(os.path.basename(polyglot_path))[0]
    # Write the words and vectors to a file
    if not os.path.exists(output_dir):
        logger.info("output dir {} does not exist, creating it".format(output_dir))
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, polyglot_filename + ".txt")

    logger.info("Writing filtered vectors to {}".format(output_path))

    with io.open(output_path, "w", encoding="utf-8") as output_file:
        for word, vector in zip(words, vectors):
            output_file.write("{} {}\n".format(
                word, " ".join(str(dim) for dim in vector)))
    logger.info("Saved vectors to {}".format(output_path))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    script_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    project_root_dir = os.path.abspath(os.path.join(script_dir,
                                                    os.pardir, os.pardir))

    parser = ArgumentParser(description=(
        "Given serialized Polyglot vectors, convert them to "
        "GloVe format and write the output file to disk"))
    parser.add_argument("--polyglot-path", metavar="<polyglot-path>", type=str,
                        help=("Path to serialized polyglot vectors to read."))
    parser.add_argument("--output-dir",
                        metavar="<output-dir>", type=str,
                        default=os.path.join(
                            project_root_dir, "data", "interim"),
                        help=("Folder to output the GloVe formatted file."))

    A = parser.parse_args()
    main(A.polyglot_path, A.output_dir)
