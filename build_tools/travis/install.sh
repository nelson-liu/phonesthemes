#!/bin/bash
set -e

echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
then
    if [[ ! -f miniconda.sh ]]
    then
        wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh \
             -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    conda create -n testenv --yes python=3.5
fi
cd ..
popd

# Activate the python environment we created.
source activate testenv

# Install requirements via pip in our conda environment
if [[ "$SKIP_TESTS" != "true" ]]; then
    pip install -r requirements.txt
else
    pip install flake8
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    # Download and setup spacy data
    echo "Downloading SpaCy data"
    python -m spacy download en

    # Set up auxiliary directories and download the GloVe and CMUDict data
    echo "Setting up auxiliary directories and downloading GloVe/CMUDict"
    python ./scripts/data/download_data.py --purge-intermediate

    # Download the Gigaword counts and the celex morphology data / segmentations,
    # and move them to the appropriate directories.
    echo "Downloading total Gigaword token counts."
    wget --quiet http://nelsonliu.me/papers/phonesthemes/data/gigaword_token_total_counts.txt
    mkdir -p ./data/processed/gigaword/
    mv gigaword_token_total_counts.txt ./data/processed/gigaword/

    echo "Downloading CELEX2 raw English morpheme data."
    wget --quiet ${CELEX_EML}
    tar -xvf eml.tar
    mkdir -p ./data/interim/celex2/english/
    mv eml ./data/interim/celex2/english/
fi

pip list
