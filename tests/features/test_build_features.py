from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from phonesthemes.features.build_features import get_ngrams
from phonesthemes.features.build_features import build_ngram_features
from phonesthemes.features.build_features import build_morpheme_features


class TestBuildFeatures(TestCase):
    def test_get_ngrams_string_word_input(self):
        answers = [[("p", "-")],
                   [("-", "s")],
                   [("p", "-"), ("-", "s")],
                   [("p", "-"), ("h",), ("o",), ("n",), ("e",), ("s",), ("t",),
                    ("h",), ("e",), ("m",), ("e",), ("-", "s",)],

                   [("p", "h", "-")],
                   [("-", "e", "s")],
                   [("p", "h", "-"), ("-", "e", "s")],
                   [("p", "h", "-"), ("h", "o"), ("o", "n"), ("n", "e"),
                    ("e", "s"), ("s", "t"), ("t", "h"), ("h", "e"),
                    ("e", "m"), ("m", "e"), ("-", "e", "s")],

                   [("p", "h", "o", "-")],
                   [("-", "m", "e", "s")],
                   [("p", "h", "o", "-"), ("-", "m", "e", "s")],
                   [("p", "h", "o", "-"), ("h", "o", "n"), ("o", "n", "e"),
                    ("n", "e", "s"), ("e", "s", "t"), ("s", "t", "h"),
                    ("t", "h", "e"), ("h", "e", "m"), ("e", "m", "e"),
                    ("-", "m", "e", "s")]]
        count = 0
        for n in [1, 2, 3]:
            for mode in [["start"], ["end"], ["start", "end"], ["all"]]:
                ngrams = get_ngrams("phonesthemes", n, mode)
                self.assertSequenceEqual(answers[count], ngrams)
                count += 1

    def test_get_ngrams_list(self):
        answers = [[("F", "-")],
                   [("-", "Z")],
                   [("F", "-"), ("-", "Z")],
                   [("F", "-"), ("OW",), ("N",), ("-", "Z")],

                   [("F", "OW", "-")],
                   [("-", "N", "Z")],
                   [("F", "OW", "-"), ("-", "N", "Z")],
                   [("F", "OW", "-"), ("OW", "N"), ("-", "N", "Z")],

                   [("F", "OW", "N", "-")],
                   [("-", "OW", "N", "Z")],
                   [("F", "OW", "N", "-"), ("-", "OW", "N", "Z")],
                   [("F", "OW", "N", "-"), ("-", "OW", "N", "Z")],

                   [("-", "F", "OW", "N", "Z", "-")],
                   [("-", "F", "OW", "N", "Z", "-")],
                   [("-", "F", "OW", "N", "Z", "-")],
                   [("-", "F", "OW", "N", "Z", "-")]]
        count = 0
        for n in [1, 2, 3, 4]:
            for mode in [["start"], ["end"], ["start", "end"], ["all"]]:
                # G2P output for "phones"
                ngrams = get_ngrams(["F", "OW", "N", "Z"], n, mode)
                self.assertSequenceEqual(answers[count], ngrams)
                count += 1

    def test_build_ngram_features(self):
        answers = [np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                             [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                   np.array([[1.0],
                             [0.0],
                             [1.0],
                             [0.0]]),
                   np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                              1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                              0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                             [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                   np.array([[1.0],
                             [0.0],
                             [1.0],
                             [0.0]])]
        vocabulary = ["hello", "goodbye", "hem", "go"]
        count = 0
        for mode in [["start"], ["all"]]:
            for freq_thres in [1, 2]:
                features, ngram_to_idx = build_ngram_features(
                    vocabulary,
                    one_hot=True,
                    ngram_range=[2, 3],
                    mode=mode,
                    freq_thres=freq_thres)
                assert_array_equal(answers[count],
                                   features)
                count += 1

    def test_build_morpheme_features(self):
        bound_morphemes_list = ["er", "waste", "fully", "ly",
                                "re", "construct", "ion", "in",
                                "un", "on"]

        word_to_morphemes = {
            "accuser": ["er"],
            "wastefully": ["waste", "fully", "ly"],
            "reconstruction": ["re", "construct", "ion"],
        }
        vocabulary = ["accuser", "wastefully", "reconstruction", "wastely"]
        features = build_morpheme_features(vocabulary, bound_morphemes_list,
                                           word_to_morphemes)

        # Verify morphemes for words in vocabulary when using dictionary
        # of words to their constituent morphemes.
        for word_idx, vector in enumerate(features):
            word = vocabulary[word_idx]
            for morpheme_idx, morpheme_val in enumerate(vector):
                if morpheme_val != 0.0:
                    if word in word_to_morphemes:
                        self.assertTrue(bound_morphemes_list[int(morpheme_idx)] in
                                        word_to_morphemes[word])
                    else:
                        self.assertTrue(bound_morphemes_list[int(morpheme_idx)]
                                        in word)
                else:
                    if word in word_to_morphemes:
                        self.assertFalse(bound_morphemes_list[int(morpheme_idx)] in
                                         word_to_morphemes[word])
                    else:
                        self.assertFalse(bound_morphemes_list[int(morpheme_idx)]
                                         in word)

        features = build_morpheme_features(vocabulary, bound_morphemes_list)

        # Verify morphemes for words in vocabulary when not using dictionary
        # of words to their constituent morphemes.
        for word_idx, vector in enumerate(features):
            word = vocabulary[word_idx]
            for morpheme_idx, morpheme_val in enumerate(vector):
                if morpheme_val != 0.0:
                    self.assertTrue(bound_morphemes_list[int(morpheme_idx)] in
                                    word)
                else:
                    self.assertFalse(bound_morphemes_list[int(morpheme_idx)] in
                                     word)
