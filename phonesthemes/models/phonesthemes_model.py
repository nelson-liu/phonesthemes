from collections import OrderedDict
import mmap
import logging
import pprint
import random

import numpy as np
import six
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, MultiTaskElasticNetCV
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ..features.build_features import build_morpheme_features
from ..features.build_features import build_ngram_features
from ..features.build_features import get_ngrams

logger = logging.getLogger(__name__)


def get_phonesthemes_from_model(model):
    if not model.is_trained:
        raise ValueError("Model must be trained before "
                         "getting phonesthemes from it.")
    logger.info("Extracting phonesthemes "
                "from ElasticNetCV model {}".format(model.phonesthemes_reg))
    logger.info("Phonestheme Features Shape: {}".format(model.X_ngram.shape))
    logger.info("Targets Shape: {}".format(np.asarray(list(
        model.vectors.values())).shape))

    # Apply the SelectFromModel meta-transformer to select
    # the most predictive subset of the features.
    phonesthemes_reg_trimmed = SelectFromModel(model.phonesthemes_reg,
                                               prefit=True)
    selected_features = phonesthemes_reg_trimmed.transform(model.X_ngram)
    logger.info("Shape of selected features: {}".format(selected_features.shape))
    # Calculate the importances of each feature.
    importances = np.sum(np.abs(model.phonesthemes_reg.coef_), axis=0)

    # Sort the feature indices by their importance.
    importance_sorted_idxs = importances.argsort()[::-1]

    # The indices chosen by SelectFromModel, but sorted
    # in order of decreasing importance.
    selected_importance_sorted_idxs = importance_sorted_idxs[
        :selected_features.shape[1]]

    # Reverse the ngram to idx dict to get a mapping from feature idx to ngram.
    idx_to_ngram = {v: k for k, v in model.ngram_to_idx.items()}

    # List of ngram features selected by SelectFromModel.
    selected_ngrams = []
    for selected_feature_index in selected_importance_sorted_idxs:
        selected_ngrams.append(idx_to_ngram[selected_feature_index])

    # Find the words containing the phonesthemes with the lowest scores in the model.
    # A list of lists containing the words with the lowest MSE for each
    # phonestheme (sorted by decreasing importance).
    all_ngram_scores = []
    for selected_ngram in selected_ngrams:
        ngram_scores = []
        original_vocabulary = list(model.vectors.keys())
        for idx, word in enumerate(original_vocabulary):
            for n in model.ngrams:
                generated_ngrams = get_ngrams(word, n, model.mode)
                if selected_ngram in generated_ngrams:
                    # Get the feature vector of the word.
                    ngram_features = model.X_ngram[idx]

                    # Make a prediction of the semantic vector from the ngram features.
                    # Reshaping data since there is just a single sample.
                    ngram_predicted_vector = model.phonesthemes_reg.predict(
                        ngram_features.reshape(1, -1))

                    # Get the true vector, reshape because there is one sample.
                    true_vector = model.vectors[word].reshape(1, -1)

                    # Calculate some error to rank the words by.
                    ngram_error = mean_squared_error(
                        true_vector,
                        ngram_predicted_vector)
                    total_error = ngram_error

                    if isinstance(word, tuple):
                        ngram_scores.append((word, model.phonemes_to_graphemes[word],
                                             total_error))
                    else:
                        ngram_scores.append((word, total_error))
        scores = sorted(ngram_scores, key=lambda x: x[-1])
        all_ngram_scores.append(scores[:10])
    return selected_ngrams, all_ngram_scores


class PhonesthemesModel(object):
    """
    Attributes
    ----------
    self.config: Dict
        A dictionary of the arguments passed into the object.

    self.ngrams: List[int]
        A list of integers that refer to the ngram sizes to use.

    self.mode: List[str]
        List of str indicating the positions in the word to use
        as candidate phonesthemes. Possible elements are "start",
        "end", and "all".

    self.min_count: int
        Minimum number of ngram occurrences in order to be included
        as a features.

    self.one_hot: bool
        Whether or not to use one-hot features instead of counts for
        the phonestheme ngram features.

    self.vectors
        Dictionary of word to vector, where word is either a string or a
        tuple of strings (phoneme representation).

    self.phonesthemes_reg
        The MultiTaskElasticNetCV model fit on the phonestheme feature vectors to
        predict the phonestheme targets.

    self.X_ngram
        The input feature vectors used to fit the Elastic Net.

    self.ngram_to_idx
        A mapping from ngram to feature index of X_ngram.

    self.is_trained
        A boolean describing whether this model has been trained or not.
    """
    def __init__(self, ngrams, mode, min_count, one_hot):
        self.config = locals()
        self.config.pop("self")
        self.config.pop("__class__", None)

        logger.info("Config: ")
        pprint.pprint(self.config)

        self.ngrams = ngrams
        self.mode = mode
        self.min_count = min_count
        self.one_hot = one_hot

        # Placeholder values, these get set when we call train
        self.vectors = None
        self.phonesthemes_reg = None
        self.X_ngram = None
        self.ngram_to_idx = None
        self.phonemes_to_graphemes = None

        self.is_trained = False

    def get_phonesthemes(self):
        return get_phonesthemes_from_model(self)

    def train(self, vectors_path, bound_morphemes_path=None,
              word_segmentations_path=None, graphemes_to_phonemes_path=None,
              n_jobs=1, l1_ratio=0.5):
        train_config = locals()
        train_config.pop("self")
        train_config.pop("__class__", None)
        self.config["train_config"] = train_config
        logger.info("Train config: ")
        pprint.pprint(train_config)

        # Load vectors, where the keys can be words represented as
        # sequences of characters (normal word vectors) or words represented
        # as sequences of phonemes (phonemicized vectors).
        logger.info("Reading vectors from {}".format(vectors_path))
        self.vectors = OrderedDict()
        with open(vectors_path) as vectors_file:
            for line in tqdm(vectors_file,
                             total=get_line_number(vectors_path)):
                split_line = line.rstrip("\n").split()
                word = split_line[0]
                # If we have phonemicized vectors, the keys to the dict are
                # tuples of comma-separated phonemes representing a word.
                if graphemes_to_phonemes_path is not None:
                    word = tuple(word.split(","))
                embedding = np.array([float(val) for val in split_line[1:]])
                self.vectors[word] = embedding

        # Randomly shuffle the OrderedDict
        random_seed = 0
        logger.info("Shuffling vectors with random seed {}".format(random_seed))
        random.seed(random_seed)
        vector_items = list(self.vectors.items())
        # random.shuffle is in-place
        random.shuffle(vector_items)
        self.vectors = OrderedDict(vector_items)

        vocabulary = list(self.vectors.keys())
        targets = np.asarray(list(self.vectors.values()))

        # Load phonemes to graphemes if we were given g2p data
        if graphemes_to_phonemes_path:
            logger.info("Reading graphemes to phonemes data "
                        "from {}".format(graphemes_to_phonemes_path))
            self.phonemes_to_graphemes = {}
            # Load the graphemes to phonemes data
            with open(graphemes_to_phonemes_path) as graphemes_to_phonemes_file:
                for line in tqdm(graphemes_to_phonemes_file,
                                 total=get_line_number(graphemes_to_phonemes_path)):
                    split_line = line.rstrip("\n").split("\t")
                    word = split_line[0]
                    phonemes = tuple(split_line[1].split(" "))
                    self.phonemes_to_graphemes[phonemes] = word

        if bound_morphemes_path is not None:
            # Load morpheme data if we were given bound morphemes
            word_segmentations, bound_morphemes = self._load_morpheme_data(
                word_segmentations_path, bound_morphemes_path)
            # Update targets with predictions of the morpheme model. This is equivalent
            # to using the model residuals as the new targets.
            targets = self._get_morpheme_residuals(vocabulary, targets, bound_morphemes,
                                                   graphemes_to_phonemes_path,
                                                   word_segmentations, n_jobs=n_jobs)

        # Get the ngram features for the vocabulary.
        self.X_ngram, self.ngram_to_idx = build_ngram_features(
            vocabulary=vocabulary,
            one_hot=self.one_hot, ngram_range=self.ngrams, mode=self.mode,
            freq_thres=self.min_count)
        logger.info("Shape of ElasticNet input (number of words, "
                    "number of candidate phonesthemes): {}".format(
                        self.X_ngram.shape))
        logger.info("Shape of ElasticNet targets (number of words, "
                    "vector dimension): {}".format(
                        targets.shape))
        # Fit a MultiTaskElasticNetCV model to extract phonesthemes.
        logger.info("Fitting MultiTaskElasticNetCV")
        self.phonesthemes_reg = MultiTaskElasticNetCV(
            l1_ratio=l1_ratio, n_jobs=n_jobs, random_state=0,
            cv=5)
        self.phonesthemes_reg.fit(self.X_ngram, targets)
        logger.info("Done fitting MultiTaskElasticNetCV")

        self.is_trained = True

    def _load_morpheme_data(self, word_segmentations_path,
                            bound_morphemes_path):
        # Load word segmentations
        word_segmentations = {}
        if word_segmentations_path:
            logger.info("Loading word segmentations from {}".format(
                word_segmentations_path))
            with open(word_segmentations_path) as word_segmentations_file:
                for line in tqdm(word_segmentations_file,
                                 total=get_line_number(word_segmentations_path)):
                    split_line = line.rstrip("\n").split("\t")
                    assert len(split_line) == 2
                    word = split_line[0]
                    morphemes = split_line[1].split(" ")
                    word_segmentations[word] = morphemes
            logger.info("Loaded {} word segmentations".format(len(word_segmentations)))

        # Load the list of bound morphemes
        logger.info("Loading bound morphemes from {}".format(bound_morphemes_path))
        bound_morphemes = []
        with open(bound_morphemes_path) as bound_morphemes_file:
            for line in tqdm(bound_morphemes_file,
                             total=get_line_number(bound_morphemes_path)):
                bound_morphemes.append(line.rstrip("\n"))
        logger.info("Loaded {} bound morphemes".format(len(bound_morphemes)))
        return (word_segmentations, bound_morphemes)

    def _get_morpheme_residuals(self, vocabulary, targets, bound_morphemes,
                                graphemes_to_phonemes_path, word_segmentations=None,
                                n_jobs=1):
        # Get the vectors vocabulary, and convert to string if we are using
        # phonemicized vectors.
        if graphemes_to_phonemes_path is None:
            string_vectors_vocab = vocabulary
        else:
            # The vocab of the phonemicized vectors converted to graphemes.
            string_vectors_vocab = [self.phonemes_to_graphemes[phonemes] for
                                    phonemes in vocabulary]
        # Build the morpheme feature vectors.
        morpheme_features = build_morpheme_features(
            string_vectors_vocab, bound_morphemes, word_segmentations)
        logger.info("Input shape for morpheme pretraining linear regression "
                    "(number of words, number of morphemes): {}".format(
                        morpheme_features.shape))
        logger.info("Target shape for morpheme pretraining linear regression "
                    "(number of words, vector dimension): {}".format(
                        targets.shape))
        morph_reg = LinearRegression(n_jobs=n_jobs)
        logger.info("Pretraining on morpheme features.")
        morph_reg = morph_reg.fit(morpheme_features, targets)
        logger.info("Calculating residuals of of linear regression done "
                    "on morpheme features and using that as the train "
                    "vectors for the ngram feature model.")

        # Get the residuals of the model for use in the second model.
        morph_reg_pred_y = morph_reg.predict(morpheme_features)
        morph_reg_residuals = np.subtract(targets, morph_reg_pred_y)
        return morph_reg_residuals

    def __eq__(self, other):
        # Two PhonesthemesModel objects are the same if their members are
        # the same.
        # Compare their ngrams
        if self.ngrams != other.ngrams:
            return False
        # Compare their mode
        if self.mode != other.mode:
            return False
        # Compare their min count
        if self.min_count != other.min_count:
            return False
        # Compare whether they use one-hot or frequency features
        if self.one_hot != other.one_hot:
            return False
        # Compare that they have the same set of vectors in the same order
        if len(self.vectors) != len(other.vectors):
            return False
        for this_word, other_word in zip(self.vectors, other.vectors):
            if this_word != other_word:
                return False
            if not np.allclose(self.vectors[this_word], other.vectors[this_word]):
                return False
        # Check that they were trained on the same features
        if not np.allclose(self.X_ngram, other.X_ngram):
            return False
        # Check that they have the same mapping of ngram to feature idx
        if self.ngram_to_idx != other.ngram_to_idx:
            return False
        return True

    if six.PY2:
        def __ne__(self, other):
            equal = self.__eq__(other)
            return equal if equal is NotImplemented else not equal


def get_line_number(file_path):
    """Fast function to calculate total number of lines in text file."""
    # from http://stackoverflow.com/a/850962/1877942
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
