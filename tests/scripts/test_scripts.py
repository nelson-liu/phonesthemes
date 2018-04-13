import hashlib
import logging
import os
import pickle
import subprocess

import numpy as np

from ..common.test_case import PhonesthemesTestCase

logger = logging.getLogger(__name__)


class TestScripts(PhonesthemesTestCase):
    def test_data_end_to_end_reproducibility(self):
        # Run clean_and_filter_vectors
        script_path = os.path.join(self.project_root_path, "scripts",
                                   "data", "clean_and_filter_vectors.py")
        output_dir = os.path.join(
            self.test_dir, "data", "processed")
        vectors_path = os.path.join(self.project_root_path, "data",
                                    "interim", "glove.840B.300d.txt")
        counts_path = os.path.join(
            self.project_root_path, "data", "processed", "gigaword",
            "gigaword_token_total_counts.txt")

        script_command = ["python", script_path,
                          "--vectors_path", vectors_path,
                          "--counts_path", counts_path,
                          "--min_token_count", "1000",
                          "--max_document_ratio", "0.5",
                          "--output_dir", output_dir]
        logger.info("Running command: {}".format(script_command))
        process = subprocess.Popen(script_command)
        process.communicate()
        return_code = process.returncode
        if return_code != 0:
            logger.warning("Got return code {} when running clean and filter "
                           "script".format(return_code))
            assert False

        filtered_vectors_hash = "d39117e503031a64ed9f0196a4bcb0c8"
        filtered_vectors_path = os.path.join(output_dir, "glove.840B.300d.filtered.txt")
        with open(filtered_vectors_path, "rb") as filtered_vectors_file:
            assert hashlib.md5(
                filtered_vectors_file.read()).hexdigest() == filtered_vectors_hash
        # Run make_phoneme_vectors
        script_path = os.path.join(self.project_root_path, "scripts",
                                   "data", "phonemicize_vectors.py")
        output_dir = os.path.join(
            self.test_dir, "data", "processed")
        filtered_vectors_path = os.path.join(output_dir, "glove.840B.300d.filtered.txt")
        g2p_path = os.path.join(self.project_root_path, "data",
                                "external", "cmudict.dict")
        script_command = ["python", script_path,
                          "--vectors_path", filtered_vectors_path,
                          "--graphemes_to_phonemes_path", g2p_path,
                          "--output_dir", output_dir]
        logger.info("Running command: {}".format(script_command))
        process = subprocess.Popen(script_command)
        process.communicate()
        return_code = process.returncode
        if return_code != 0:
            logger.warning("Got return code {} when running clean and filter "
                           "script".format(return_code))
            assert False

        phonemicized_vectors_hash = "21b431731467fdc1ac3ef887549f7e5f"
        phonemicized_vectors_path = os.path.join(
            output_dir, "glove.840B.300d.filtered.phonemicized.txt")
        with open(phonemicized_vectors_path, "rb") as phonemicized_vectors_file:
            actual_hash = hashlib.md5(
                phonemicized_vectors_file.read()).hexdigest()
            assert actual_hash == phonemicized_vectors_hash

        g2p_hash = "67734616f17c7721872332bf7881100f"
        g2p_path = os.path.join(
            output_dir, "glove.840B.300d.filtered.graphemes_to_phonemes.txt")
        with open(g2p_path, "rb") as g2p_file:
            assert hashlib.md5(
                g2p_file.read()).hexdigest() == g2p_hash

    def test_clean_and_filter_vectors(self):
        self.write_vector_file()
        self.write_vector_frequencies_file()
        script_path = os.path.join(self.project_root_path, "scripts",
                                   "data", "clean_and_filter_vectors.py")

        output_dir = os.path.join(
            self.test_dir, "data", "processed")
        script_command = ["python", script_path,
                          "--vectors_path", self.vectors_path,
                          "--counts_path", self.vectors_counts_path,
                          "--min_token_count", "3",
                          "--max_document_ratio", "0.5",
                          "--output_dir", output_dir]

        process = subprocess.Popen(script_command)
        process.communicate()
        return_code = process.returncode
        if return_code != 0:
            logger.warning("Got return code {} when running clean and filter "
                           "script".format(return_code))
            assert False

        # Read the output filtered vectors.
        correct_dict = {"wordone": np.array([0.1, 0.4, -4.0]),
                        "wordtwo": np.array([0.0, 1.1, 0.2])}
        with open(os.path.join(
                output_dir, "vectors_file.filtered.txt")) as filtered_vectors_file:
            filtered_vectors = {}
            for line in filtered_vectors_file:
                split_line = line.split()
                word = split_line[0]
                embedding = [float(val) for val in split_line[1:]]
                filtered_vectors[word] = np.array(embedding)
        np.testing.assert_equal(correct_dict, filtered_vectors)

    def test_model_does_not_crash_and_is_reproducible(self):
        self.write_phonemicized_vector_files()
        self.write_morpheme_data()
        script_path = os.path.join(self.project_root_path, "scripts",
                                   "run", "run_model.py")
        for run_id in ["unittest_1", "unittest_2"]:
            script_command = [
                "python", script_path,
                "--ngrams", "2", "3",
                "--vectors_path", self.phonemicized_vectors_path,
                "--mode", "start",
                "--graphemes_to_phonemes_path", self.phonemicized_vectors_g2p_path,
                "--bound_morphemes_path", self.bound_morphemes_path,
                "--word_segmentations_path", self.word_segmentations_path,
                "--min_count", "1",
                "--one_hot",
                "--save_dir", self.models_path,
                "--n_jobs", "2",
                "--run_id", run_id]
            logger.info("Running command: {}".format(script_command))
            process = subprocess.Popen(script_command)
            process.communicate()
            return_code = process.returncode
            if return_code != 0:
                logger.warning("Got return code {} when running clean and filter "
                               "script".format(return_code))
                assert False

        # Load both of the writen models and assert they are equal
        run_1_path = os.path.join(self.models_path, "unittest_1",
                                  "PhonesthemesModel_unittest_1.pkl")
        run_1_model = pickle.load(open(run_1_path, "rb"))

        run_2_path = os.path.join(self.models_path, "unittest_2",
                                  "PhonesthemesModel_unittest_2.pkl")
        run_2_model = pickle.load(open(run_2_path, "rb"))
        assert run_1_model == run_2_model

        run_1_phonesthemes = run_1_model.get_phonesthemes()[0]
        run_2_phonesthemes = run_2_model.get_phonesthemes()[0]
        assert run_1_phonesthemes == run_2_phonesthemes

        # Make sure loading the model with the script works
        script_command = [
            "python", script_path,
            "--load_path", run_1_path]
        logger.info("Running command: {}".format(script_command))
        process = subprocess.Popen(script_command)
        process.communicate()
        return_code = process.returncode
        if return_code != 0:
            logger.warning("Got return code {} when running clean and filter "
                           "script".format(return_code))
            assert False

    def test_get_bound_morphemes_from_celex(self):
        script_path = os.path.join(self.project_root_path, "scripts",
                                   "data", "get_bound_morphemes.py")
        output_dir = os.path.join(
            self.test_dir, "data", "processed")
        celex_morph_data_path = os.path.join(
            self.project_root_path, "data", "interim", "celex2", "english", "eml")
        column = "22"

        script_command = ["python", script_path,
                          "--celex_morph_data_path", celex_morph_data_path,
                          "--column", column,
                          "--output_dir", output_dir]
        logger.info("Running command: {}".format(script_command))
        process = subprocess.Popen(script_command)
        process.communicate()
        return_code = process.returncode
        if return_code != 0:
            logger.warning("Got return code {} when running clean and filter "
                           "script".format(return_code))
            assert False

        segmentation_dictionary_hash = "7914f2f8dac46d791205a50b3b58e10d"
        segmentation_dictionary_path = os.path.join(
            output_dir, "eml.seg")
        with open(segmentation_dictionary_path, "rb") as segmentation_dictionary_file:
            assert (hashlib.md5(segmentation_dictionary_file.read()).hexdigest() ==
                    segmentation_dictionary_hash)
        bound_morphemes_hash = "be023287d8d3ec4183d920530fae718b"
        bound_morphemes_path = os.path.join(
            output_dir, "eml.bound_morphemes.txt")
        with open(bound_morphemes_path, "rb") as bound_morphemes_file:
            assert hashlib.md5(
                bound_morphemes_file.read()).hexdigest() == bound_morphemes_hash
