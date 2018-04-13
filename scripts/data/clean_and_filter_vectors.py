from collections import OrderedDict, Counter
import logging
import mmap
import os
import operator

from argparse import ArgumentParser
import numpy as np
import spacy
from tqdm import tqdm


def main(vectors_path, output_dir, counts_path=None, min_token_count=None,
         max_document_ratio=None, filter_lemmas=False, language="en"):
    if counts_path:
        word_counts = load_word_counts(counts_path, max_document_ratio)
    else:
        word_counts = None
    # Load the vectors from the text file, and create a dictionary of
    # {word:numpy array of vector} for words that occur in the English
    # dictionary.
    filtered_vectors = load_and_filter_vectors(
        vectors_path, word_counts, min_token_count, filter_lemmas, language)

    vector_filename = os.path.splitext(os.path.basename(vectors_path))[0]

    # Write the dictionary of word / lemma to vector to a file
    if not os.path.exists(output_dir):
        logger.info("output dir {} does not exist, creating it".format(output_dir))
        os.makedirs(output_dir)

    filtered_vectors_path = os.path.join(
        output_dir, vector_filename + ".filtered.txt")
    logger.info("Writing filtered vectors to {}".format(filtered_vectors_path))
    # Convert the dictionary to a list of tuples for reproducibility
    sorted_filtered_vectors = sorted(
        filtered_vectors.items(), key=operator.itemgetter(0))
    with open(filtered_vectors_path, "w") as filtered_vectors_file:
        for key, vector in sorted_filtered_vectors:
            filtered_vectors_file.write("{} {}\n".format(
                key, " ".join(str(dim) for dim in vector)))
    logger.info("Saved vectors to {}".format(filtered_vectors_path))


def load_word_counts(counts_path, max_document_ratio):
    logger.info("Reading token counts "
                "from {}".format(counts_path))
    word_counts = Counter()
    with open(counts_path) as freq_file:
        for line in tqdm(freq_file,
                         total=get_line_number(counts_path)):
            word, count, document_ratio = line.strip().split("\t")
            document_ratio = eval(document_ratio)
            if (not word.isalpha() or not word.islower() or len(word) <= 2 or
                    document_ratio > max_document_ratio):
                # If the word is not alphabetic, or the word is of length
                # two or less, just discard it.
                continue
            word_counts[word] += int(count)
    logger.info("Loaded {} token counts".format(len(word_counts)))
    return word_counts


def load_and_filter_vectors(vectors_path, word_counts, min_token_count,
                            filter_lemmas, language):
    if word_counts:
        logger.info(("Loading vectors from {}, keeping only those "
                     "that correspond to words occuring at least {} times "
                     "(as noted in the counts_path "
                     "file)").format(vectors_path, min_token_count))
    else:
        logger.info("Loading vectors from {}, not doing any "
                    "frequency filtering".format(vectors_path))
    nlp = spacy.load(language, disable=["parser", "ner", "textcat"])
    with open(vectors_path) as vectors_file:
        # Dictionary of word to vector
        vectors = OrderedDict()
        # Dictionary of word to lemma
        vector_words_to_lemmas = OrderedDict()
        for line in tqdm(vectors_file,
                         total=get_line_number(vectors_path)):
            split_line = line.split()
            word = split_line[0]
            if (word.isalpha() and
                    word.islower() and
                    len(word) > 2):
                if word_counts is not None:
                    if word_counts[word] < min_token_count:
                        continue
                word_lemma = nlp(word)[0].lemma_
                embedding = np.array([float(val) for val in split_line[1:]])
                vectors[word] = embedding
                vector_words_to_lemmas[word] = word_lemma
    logger.info("Loaded {} words.".format(len(vectors)))

    if filter_lemmas:
        # If two words have the same lemma, and that
        # lemma is in the set of vectors, then the two words are filtered
        # out. If the lemma is not in the set of vectors, the two words are
        # not filtered out and their unlemmatized forms are used.
        logger.info("Filtering the vectors to remove words that have the "
                    "same lemma, if the lemma also has a word vector.")
        lemma_vectors = {}
        for word in tqdm(vectors):
            lemmatized_word = vector_words_to_lemmas[word]
            if lemmatized_word in vectors:
                lemma_vectors[lemmatized_word] = vectors[lemmatized_word]
            else:
                lemma_vectors[word] = vectors[word]

        logger.info("Loaded {} {} word lemmas.".format(
            len(lemma_vectors), language))
        vectors = lemma_vectors
    return vectors


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
        "Given a text file of input vectors, generate "
        "a text file of filtered vectors."))
    parser.add_argument("--vectors_path", metavar="<vectors_path>", type=str,
                        help=("Path to the word vectors to read."))
    parser.add_argument("--counts_path",
                        metavar="<counts_path>", type=str,
                        help=("Path to file with tab-separated "
                              "word counts."))
    parser.add_argument("--min_token_count",
                        metavar="<min_token_count>", type=int,
                        default=1000,
                        help=("The lowest frequency (as given by the "
                              "counts_path file) that a token "
                              "must have in order to be kept."))
    parser.add_argument("--max_document_ratio",
                        metavar="<max_document_ratio>", type=float,
                        default=0.5,
                        help=("The highest document occurence ratio (num documents that "
                              "a token appears in / num total documents) as given by the "
                              "counts_path file that a token must have in order to "
                              "be kept."))
    parser.add_argument("--no-lemma-filtering", action="store_true",
                        help=("If set, do not remove words that have the same lemma."))
    parser.add_argument("--spacy-language", type=str, default="en",
                        help=("The code for the SpaCy language to use."))
    parser.add_argument("--output_dir",
                        metavar="<output_dir>", type=str,
                        default=os.path.join(
                            project_root_dir, "data", "processed"),
                        help=("Folder to output the file of "
                              "<comma separated phonemes><space>"
                              "<space-delimited numpy array>"))

    A = parser.parse_args()
    main(A.vectors_path, A.output_dir, A.counts_path,
         A.min_token_count, A.max_document_ratio, not A.no_lemma_filtering,
         A.spacy_language)
