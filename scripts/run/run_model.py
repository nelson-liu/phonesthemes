from argparse import ArgumentParser
import json
from os import path, pardir, makedirs
import logging
try:
    import cPickle as pickle
except:
    import pickle
import sys

sys.path.append(path.join(path.dirname(__file__), pardir, pardir))
from phonesthemes.models.phonesthemes_model import get_phonesthemes_from_model
from phonesthemes.models.phonesthemes_model import PhonesthemesModel

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    parser = ArgumentParser(
        description=("Train a linear model (MultiTaskElasticNetCV) "
                     "to predict word vectors given character-level "
                     "features from words."))
    parser.add_argument("--load_path", metavar="<load_path>",
                        type=str, help=("Load a model and output the "
                                        "predicted phonesthemes."))
    parser.add_argument("--ngrams", nargs='+', metavar="<ngrams>",
                        type=int,
                        help=("Lengths of phoneme or character ngrams "
                              "to use as features"))
    parser.add_argument("--vectors_path", metavar="<vectors_path>",
                        type=str, help=("Path to vectors file."))
    parser.add_argument("--mode", nargs="+", type=str,
                        help=("The ngrams to use as features. Possible "
                              "elements are \"start\" to indicate returning "
                              "ngram at very beginning of the token, \"end\" "
                              "to indicate returning the ngram at the very "
                              "end of the token, or \"all\" to indicate "
                              "returning all ngrams. The \"all\" mode entails "
                              "\"start\" and \"end\"."))
    parser.add_argument("--graphemes_to_phonemes_path", type=str,
                        metavar="<graphames_to_phonemes_path>",
                        help=("Path to a file with words and their "
                              "phonemicized representations. Required if "
                              "using phoneme vectors."))
    parser.add_argument("--bound_morphemes_path",
                        metavar="<bound_morphemes_path>", type=str,
                        help=("Path to the list of bound morphemes. "
                              "Required if training with morphemes."))
    parser.add_argument("--word_segmentations_path",
                        metavar="<word_segmentations_path>", type=str,
                        help=("Path to file with string words and their constituent "
                              "morphemes. Not required, but if the data will be used "
                              "to featurize each word if pretraining with morpheme "
                              "features."))
    parser.add_argument("--min_count", metavar="<min_count>",
                        type=int, help=("Minimum number of ngram occurrences in order "
                                        "to be included as a feature."))
    parser.add_argument("--one_hot", action="store_true",
                        help=("Use one-hot features instead of raw counts."))
    parser.add_argument("--save_dir", type=str,
                        help=("Directory to write serialized models. "
                              "If not specified, models are not saved."))
    parser.add_argument("--l1_ratio", type=float, nargs="+", default=0.5,
                        help=("ElasticNet L1 ratios to try --- the best "
                              "will be picked by k-fold cross-validation."))
    parser.add_argument("--n_jobs", type=int, default=1,
                        help=("Number of processes to use."))
    parser.add_argument("--run_id", type=str,
                        help=("Identifying run ID for this experiment."))

    A = parser.parse_args()

    if A.load_path:
        save_dir = path.dirname(A.load_path)
        # Load a trained model and output the predicted phonesthemes
        logger.info("Loading model from {}".format(A.load_path))
        with open(A.load_path, "rb") as load_file:
            loaded_model = pickle.load(load_file)
        selected_ngrams, all_ngram_scores = get_phonesthemes_from_model(loaded_model)

        selected_ngrams_path = path.join(save_dir, "selected_ngrams.txt")
        logger.info("Writing list of selected ngrams to {}".format(
            selected_ngrams_path))
        with open(selected_ngrams_path, 'w') as selected_ngrams_file:
            for selected_ngram in selected_ngrams:
                selected_ngrams_file.write("{}\n".format(selected_ngram))

        ngram_top_words_path = path.join(save_dir,
                                         "selected_ngrams_top_words.txt")
        logger.info("Writing list of top words for each "
                    "selected ngrams to {}".format(ngram_top_words_path))
        with open(ngram_top_words_path, "w") as top_words_file:
            for selected_ngram_top_words in all_ngram_scores:
                for top_word in selected_ngram_top_words:
                    top_words_file.write("{}\n".format(top_word))
                top_words_file.write("-" * 79)
                top_words_file.write("\n")
    else:
        if not (A.ngrams and A.vectors_path and A.mode and A.run_id):
            parser.error(
                "--ngrams, --vectors_path, --mode, and --run_id "
                "are required if training a model.")

        model = PhonesthemesModel(
            ngrams=A.ngrams,
            mode=A.mode,
            min_count=A.min_count,
            one_hot=A.one_hot)
        model.train(vectors_path=A.vectors_path,
                    bound_morphemes_path=A.bound_morphemes_path,
                    word_segmentations_path=A.word_segmentations_path,
                    graphemes_to_phonemes_path=A.graphemes_to_phonemes_path,
                    n_jobs=A.n_jobs, l1_ratio=A.l1_ratio)

        if A.save_dir:
            save_path = path.join(A.save_dir, A.run_id)
            logger.info("Saving model to {}.".format(save_path))

            if not path.exists(save_path):
                logger.info("save path {} does not exist, "
                            "creating it".format(save_path))
                makedirs(save_path)

            model_name = "PhonesthemesModel_{}.pkl".format(A.run_id)

            model_save_path = path.join(save_path, model_name)

            with open(model_save_path, "wb") as model_save_file:
                pickle.dump(model, model_save_file)
            logger.info("Wrote model to {}".format(model_save_path))

            logger.info("Saving config to disk.")
            config_save_path = path.join(save_path, "config.json")
            with open(config_save_path, "w") as config_save_file:
                json.dump(vars(A), config_save_file,
                          indent=4, sort_keys=True)
            logger.info("Wrote config to {}".format(config_save_path))
