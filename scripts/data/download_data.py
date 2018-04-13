from argparse import ArgumentParser
import hashlib
import logging
import os
import subprocess
import sys
import zipfile

import requests

logger = logging.getLogger(__name__)


GLOVE_ZIP_URL = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_ZIP_MD5 = "2ffafcc9f9ae46fc8c95f32372976137"
GLOVE_UNZIPPED_MD5 = "eec7d467bccfa914726b51aac484d43a"

CMUDICT_URL = ("https://raw.githubusercontent.com/cmusphinx/cmudict/"
               "132be0d63ec0a6179d860114d7604e315541d94a/cmudict.dict")
CMUDICT_MD5 = "ac7b83b417327fa355022c97e7566080"


def create_auxiliary_dirs():
    logger.info("Making auxiliary directories for data and saved "
                "models (not overwriting if they exist).")
    data_dirs = ["external", "interim", "processed", "raw"]
    for data_dir in data_dirs:
        os.makedirs(os.path.join(project_root_dir, "data", data_dir),
                    exist_ok=True)
    os.makedirs(os.path.join(project_root_dir, "models"), exist_ok=True)


def fetch_and_unzip_vector_data():
    glove_save_path = os.path.join(project_root_dir, "data",
                                   "external", "glove.840B.300d.zip")
    # Download the GloVe data if applicable
    if (os.path.exists(glove_save_path) and
            hashlib.md5(
                open(glove_save_path,
                     "rb").read()).hexdigest() == GLOVE_ZIP_MD5):
        logger.info("GloVe data already exists, skipping download.")
    else:
        logger.info("Downloading GloVe data to {}".format(glove_save_path))
        try:
            args = ["wget", "-O", glove_save_path, GLOVE_ZIP_URL]
            output = subprocess.Popen(args, stdout=subprocess.PIPE)
            out, err = output.communicate()
        except:
            logger.info("Couldn't download GloVe data with wget, "
                        "falling back to (slower) Python downloading.")
            glove_response = requests.get(GLOVE_ZIP_URL, stream=True)
            with open(glove_save_path, "wb") as glove_file:
                for chunk in glove_response.iter_content(chunk_size=1024 * 1024):
                    # Filter out keep-alive new chunks.
                    if chunk:
                        glove_file.write(chunk)

    # Extract the GloVe data if it does not already exist.
    glove_unzip_folder = os.path.join(project_root_dir, "data",
                                      "interim")
    glove_unzip_path = os.path.join(glove_unzip_folder, "glove.840B.300d.txt")
    if (os.path.exists(glove_unzip_path) and
            hashlib.md5(
                open(glove_unzip_path,
                     "rb").read()).hexdigest() == GLOVE_UNZIPPED_MD5):
        logger.info("Unzipped GloVe data already exists, skipping unzip.")
    else:
        logger.info("Unzipping GloVe archive to {}".format(glove_unzip_path))
        zip_ref = zipfile.ZipFile(glove_save_path, "r")
        zip_ref.extractall(glove_unzip_folder)
        zip_ref.close()


def fetch_cmudict():
    cmudict_save_path = os.path.join(project_root_dir, "data",
                                     "external", "cmudict.dict")
    # Download the GloVe data if applicable.
    if (os.path.exists(cmudict_save_path) and
            hashlib.md5(
                open(cmudict_save_path,
                     "rb").read()).hexdigest() == CMUDICT_MD5):
        logger.info("CMU Dict data already exists, skipping download.")
    else:
        logger.info("Downloading CMU Dict data to {}".format(cmudict_save_path))
        try:
            args = ["wget", "-O", cmudict_save_path, CMUDICT_URL]
            output = subprocess.Popen(args, stdout=subprocess.PIPE)
            out, err = output.communicate()
        except:
            logger.info("Couldn't download CMUDict data with wget, "
                        "falling back to (slower) Python downloading.")
            cmudict_response = requests.get(CMUDICT_URL)
            with open(cmudict_save_path, "wb") as cmudict_file:
                cmudict_file.write(cmudict_response.content)


if __name__ == "__main__":
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO, handlers=handlers)
    parser = ArgumentParser(description=(
        "Download data used in phonesthemes extraction."))
    parser.add_argument("--purge-intermediate",
                        action="store_true",
                        help=("Delete intermediate downloaded archives, "
                              "leaving only final data files."))
    args = parser.parse_args()

    script_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    project_root_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    create_auxiliary_dirs()
    fetch_and_unzip_vector_data()
    if args.purge_intermediate:
        glove_save_path = os.path.join(
            project_root_dir, "data", "external", "glove.840B.300d.zip")
        os.remove(glove_save_path)
    fetch_cmudict()
