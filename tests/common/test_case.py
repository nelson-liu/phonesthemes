import logging
import os
import shutil
from unittest import TestCase


class PhonesthemesTestCase(TestCase):
    # Path to directory this file is in.
    base_test_case_dir = os.path.abspath(os.path.dirname(
        os.path.realpath(__file__)))
    # Project root path
    project_root_path = os.path.abspath(os.path.join(
        base_test_case_dir, os.pardir, os.pardir))
    test_dir = "./TMP_TEST/"
    vectors_path = os.path.join(test_dir, "vectors_file")
    vectors_counts_path = os.path.join(test_dir, "vectors_counts_file")
    phonemicized_vectors_path = os.path.join(test_dir, "phonemicized_vectors_file")
    phonemicized_vectors_g2p_path = os.path.join(test_dir, "phonemicized_vectors_g2p")
    bound_morphemes_path = os.path.join(test_dir, "bound_morphemes")
    word_segmentations_path = os.path.join(test_dir, "word_segmentations")
    models_path = os.path.join(test_dir, "models")

    def setUp(self):
        logging.basicConfig(format=("%(asctime)s - %(levelname)s - "
                                    "%(name)s - %(message)s"),
                            level=logging.INFO)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def write_vector_file(self):
        with open(self.vectors_path, "w") as vectors_file:
            vectors_file.write("word1 0.1 0.4 -4.0\n")
            vectors_file.write("word2 0.0 1.1 0.2\n")
            vectors_file.write("wordone 0.1 0.4 -4.0\n")
            vectors_file.write("wordtwo 0.0 1.1 0.2\n")
            vectors_file.write("otherword, 0.2, 0.3, 0.4\n")

    def write_vector_frequencies_file(self):
        with open(self.vectors_counts_path, "w") as vectors_freq_file:
            vectors_freq_file.write("wordone\t5\t30/100\n")
            vectors_freq_file.write("wordtwo\t3\t40/100\n")
            vectors_freq_file.write("otherword\t2\t45/100\n")
            vectors_freq_file.write("ignoredword\t2\t60/100\n")

    def write_phonemicized_vector_files(self):
        with open(self.phonemicized_vectors_path, "w") as pho_vectors_file:
            pho_vectors_file.write("EH0,S,TH,EH1,T,IH0,K 0.1 0.4 -4.0\n")
            pho_vectors_file.write("F,UW1,L,IH0,SH,N,AH0,S 0.0 1.1 0.2\n")
            pho_vectors_file.write("F,AH1,N,AH0,L,Z 0.1 0.4 -4.0\n")
            pho_vectors_file.write("D,UW1 0.0 1.1 0.2\n")
            pho_vectors_file.write("L,AA1,K,S,T,EH2,P 0.2 0.3 0.4\n")

        with open(self.phonemicized_vectors_g2p_path, "w") as g2p_file:
            g2p_file.write("esthetic\tEH0 S TH EH1 T IH0 K\n")
            g2p_file.write("foolishness\tF UW1 L IH0 SH N AH0 S\n")
            g2p_file.write("funnels\tF AH1 N AH0 L Z\n")
            g2p_file.write("dew\tD UW1\n")
            g2p_file.write("lockstep\tL AA1 K S T EH2 P\n")

    def write_morpheme_data(self):
        with open(self.bound_morphemes_path, "w") as bound_morphemes_file:
            bound_morphemes_file.write("ness\n")
            bound_morphemes_file.write("s\n")
            bound_morphemes_file.write("ish\n")
            bound_morphemes_file.write("ion")

        with open(self.word_segmentations_path, "w") as word_segmentations_file:
            word_segmentations_file.write("foolishness\tfool ish ness\n")
            word_segmentations_file.write("funnels\tfunnel s\n")
            word_segmentations_file.write("lockstep\tlock step")
