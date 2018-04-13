from collections import Counter
import numpy as np
import logging

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def build_ngram_features(vocabulary, one_hot, ngram_range,
                         mode, freq_thres=1):
    """
    Build ngram features for phonestheme extraction.

    Parameters
    ----------
    vocabulary : list
        The list of tokens to create features from. If the input is a string,
        the ngrams are character-level (treating the string as a list of
        characters).

    one_hot : boolean
        If True, feature vectors are one-hot that indicate the
        presence of a ngram.

    ngram_range : list
        A list of the size of ngrams to use as features.

    mode : list
        A list of string indicating the features to use. Possible
        elements are "start" to indicate using the ngram at the very beginning
        of the token, "end" to indicate using the ngram at the very end
        of the token, or "all" to indicate using all ngrams. The "all" mode
        entails "start" and "end".

    freq_thres : int, optional, default=1
        Removes any ngrams features that occur less than
        `freq_thres` times across the whole vocabulary. By
        default, it does not remove any ngrams.

    Returns
    -------
    X : numpy.ndarray
        The numpy array (2D array) X contains
        features derived from the input `vocabulary` of shape
        `(len(vocabulary), num_features)`. The number of features is the
        number of unique ngrams in the corpus.
    """
    # get ngrams from all words
    logger.info("Building ngram feature vectors.")
    logger.info("Extracting unique n-grams (of length {}) "
                "from corpus. mode={}".format(ngram_range,
                                              mode))
    # a dictionary of ngrams to the number of
    # times they occur in the corpus
    all_ngram_counts = Counter()

    for word in vocabulary:
        for n in ngram_range:
            for ngram in get_ngrams(word, n, mode):
                all_ngram_counts[ngram] += 1

    logger.info("Extracted {} ngrams of length {} from corpus".format(
        len(all_ngram_counts), ngram_range))

    # a dictionary of ngram to ngram index that only
    # has the ngrams that occur at least freq_thres times.
    freq_filtered_ngrams = {}
    # filter the ngrams based on the number of times they occur
    # We sort the keys here to keep iteration order consistent
    # across runs, which enhances reproducibility.
    for ngram in sorted(list(all_ngram_counts.keys())):
        if all_ngram_counts[ngram] >= freq_thres:
            freq_filtered_ngrams[ngram] = len(freq_filtered_ngrams)
    logger.info("Filtered ngrams that occur less than {} times, number of ngrams "
                "post-filtering: {}".format(freq_thres, len(freq_filtered_ngrams)))

    logger.info("Creating ngram feature vectors for each "
                "word. one_hot={}".format(one_hot))
    # create features for each of the words
    # Generate an empty feature array that we will populate.
    # The number of rows is the number of words in the vocabulary,
    # and the number of columns is the number of features
    # (num ngrams post filtering).
    X = np.zeros((len(vocabulary), len(freq_filtered_ngrams)))
    # iterate through the words in the vocabulary
    for idx, word in enumerate(vocabulary):
        # Generate the ngrams for each size in ngram_range
        for n in ngram_range:
            for ngram in get_ngrams(word, n, mode):
                # if the ngram generated is in the set of
                # features post-filtering, add it to the feature matrix.
                if ngram in freq_filtered_ngrams:
                    X[idx][freq_filtered_ngrams[ngram]] += 1

    # if we want a one_hot vector, set everything that is greater
    # than 0 in the feature matrix to be 1.
    if one_hot:
        X[X > 0] = 1
    return X, freq_filtered_ngrams


def build_morpheme_features(vocabulary, bound_morphemes,
                            word_morpheme_dict=None):
    """
    Build morpheme features for phonestheme induction.
    Outputs a one-hot vector with indexes set to 1 if
    the morpheme is in the word, where in is either defined by
    substring or if it is in the list of the word's constituent
    morphemes.

    Parameters
    ----------
    vocabulary : list
        The list of words to create morpheme features of.

    bound_morphemes : list
        A list of strings, where each string is a bound morpheme.

    word_morpheme_dict: dict of {str: List[str]}, optional (default=None)
        A dict of from the string representation of the word to a list
        of strings, where the list of strings contains the constituent
        morphemes of the word. If we want to get the features for a word
        that is not in this dictionary, we fall back to substring.

    Returns
    -------
    X : numpy.ndarray
        The ``ndarray`` matrix ``X`` contains
        features derived from the input ``vocabulary`` of shape
        ``(len(vocabulary), num_features)``. The number of features is
        the number of input bound morphemes.
    """

    # get ngrams from all words
    logger.info("Creating morpheme feature vectors for each word.")
    # create morpheme features for each of the words by setting the
    # appropriate index in the one hot vector to one if the morpheme
    # is in the word.
    X = np.zeros((len(vocabulary), len(bound_morphemes)))
    # Iterate over every word in the vocabulary
    for word_idx, word in enumerate(vocabulary):
        # For each word, iterate over the list of bound morphemes.
        for morpheme_idx, morpheme in enumerate(bound_morphemes):
            # If the bound morpheme is in the word, then set that
            # feature to 1.

            if word_morpheme_dict and word in word_morpheme_dict:
                # If the word is in the word_morpheme_dict,
                # then take that to be the truth.
                if morpheme in word_morpheme_dict[word]:
                    X[word_idx][morpheme_idx] = 1
            else:
                # If the word is not in the word_morpheme_dict, or we were
                # not provided a word_morpheme_dict, fall back to substring.
                if morpheme in word:
                    X[word_idx][morpheme_idx] = 1
    return X


def get_ngrams(word, n, mode):
    """
    Get character ngrams from a word.

    Parameters
    ----------

    word : String
        The word to extract character n-grams from.

    n : int
        Length of each n-gram to extract. If ``n`` is greater
        than the length of ``word``, then an empty list is returned.

    mode : list
        A list of string indicating the ngrams to return. Possible
        elements are "start" to indicate returning ngram at the very beginning
        of the token, "end" to indicate returning the ngram at the very end
        of the token, or "all" to indicate returning All ngrams. The "all" mode
        entails "start" and "end".

    Returns
    -------

    ngrams_to_return : list of tuples
        A list of tuples of ngrams extracted from the input ``word``.
    """
    if n > len(word):
        return []
    if not isinstance(word, list):
        word = list(word)
    ngrams_to_return = []
    ngrams = list(zip(*[word[i:] for i in range(n)]))
    # add "-" after first ngram to show that it is first
    first_ngram = list(ngrams[0])
    first_ngram.append("-")
    ngrams[0] = tuple(first_ngram)

    # add "-" before last ngram to show that it is the last
    # if ngrams[-1] != ngrams[0] and len(ngrams) != 1:
    last_ngram = list(ngrams[-1])
    last_ngram.insert(0, "-")
    ngrams[-1] = tuple(last_ngram)

    if "all" in mode:
        return ngrams
    if "start" in mode:
        ngrams_to_return.append(ngrams[0])
    if "end" in mode:
        ngrams_to_return.append(ngrams[-1])
    # if there is only one ngram for start and end case
    if ngrams[-1] == ngrams[0]:
        return [ngrams_to_return[0]]
    return ngrams_to_return
