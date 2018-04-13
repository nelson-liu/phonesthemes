from collections import Counter
import logging
from functools import partial
import os
import multiprocessing
import spacy

from argparse import ArgumentParser
from bs4 import BeautifulSoup
from tqdm import tqdm

en_nlp = spacy.load("en")


def main(gigaword_path, output_counts_path, count_dateline, num_processes):
    logger.info("Reading all data files in the Gigaword "
                "directory at {}.".format(gigaword_path))
    logger.info("Count dateline: {}".format(count_dateline))
    gigaword_data_directory = os.path.join(gigaword_path, "data")
    file_list = []
    for root, subFolders, files in os.walk(gigaword_data_directory):
        for news_file in files:
            file_list.append(os.path.join(root, news_file))

    logger.info("Reading {} files with {} processes".format(len(file_list),
                                                            num_processes))

    pool = multiprocessing.Pool(num_processes)

    # Map the files into a list of result tuples
    partial_process_one_gigaword_file = partial(process_one_gigaword_file,
                                                count_dateline=count_dateline)
    total_counts = Counter()
    document_counts = Counter()
    try:
        for result_tuple in tqdm(pool.imap_unordered(partial_process_one_gigaword_file,
                                                     file_list),
                                 total=len(file_list)):
            # tokenized_words_in_file is a list of strings, representing
            # the document after it has been tokenized.
            # file_vocab is a set of all of the words in the document
            tokenized_words_in_file, file_vocab = result_tuple
            total_counts.update(tokenized_words_in_file)
            document_counts.update(file_vocab)
    except KeyboardInterrupt:
        logging.warning("Got Ctrl+C")
    finally:
        pool.terminate()
        pool.join()
    num_documents = len(file_list)
    logger.info("Read {} documents, counted {} total tokens.".format(
        num_documents, len(total_counts)))
    logger.info("Sorting the counts by increasing count order, then "
                "by alphabetical order for ties.")
    sorted_words_and_counts = sorted(total_counts.items(),
                                     key=lambda pair: (-pair[1], pair[0]))
    logger.info("Done sorting.")

    logger.info("Writing tokens, token counts, and token document frequencies "
                "as a text file to {}".format(output_counts_path))
    with open(output_counts_path, "w") as output_file:
        for word, count in sorted_words_and_counts:
            output_file.write("{}\t{}\t{}/{}\n".format(
                word, count, document_counts[word], num_documents))
    logger.info("Wrote token, token counts, and token document "
                "frequencies to {}".format(output_counts_path))


def process_one_gigaword_file(file_path, count_dateline):
    # Parse the text with BeautifulSoup
    soup = BeautifulSoup(open(file_path), "html.parser")

    # remove the DATELINE elements if we are not counting datelines.
    if not count_dateline:
        for dateline in soup.find_all("dateline"):
            dateline.extract()

    # get the raw text
    text = soup.get_text()

    # tokenize the text and put it into a list
    tokens = en_nlp.tokenizer(text)
    raw_tokens_list = [str(token) for token in tokens if not
                       str(token).isspace()]
    token_vocab_in_file = set(raw_tokens_list)
    return (raw_tokens_list, token_vocab_in_file)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    parser = ArgumentParser(description=("Get the frequencies for all words "
                                         "in the Gigaword corpus."))
    parser.add_argument("--gigaword_path", required=True,
                        metavar="<gigaword_path>", type=str,
                        help=("Path to Gigaword directory, with "
                              "all .gz files unzipped."))
    parser.add_argument("--count_dateline", action="store_true",
                        help=("If this flag is set, the dateline is not "
                              "removed from the gigaword files."))
    parser.add_argument("--num_processes", type=int, default=1,
                        help=("Number of processes to use to read "
                              "the files in parallel"))
    parser.add_argument("--output_counts_path", required=True,
                        metavar="<output_counts_path>", type=str,
                        help=("Path to output a text file "
                              "with frequency data."))

    A = parser.parse_args()
    main(A.gigaword_path, A.output_counts_path, A.count_dateline, A.num_processes)
