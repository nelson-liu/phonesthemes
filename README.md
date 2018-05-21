[![Build Status](https://travis-ci.org/nelson-liu/phonesthemes.svg?branch=master)](https://travis-ci.org/nelson-liu/phonesthemes)
[![codecov](https://codecov.io/gh/nelson-liu/phonesthemes/branch/master/graph/badge.svg)](https://codecov.io/gh/nelson-liu/phonesthemes)

# Discovering Phonesthemes with Sparse Regularization

Phonestheme induction and discovery through sparse feature selection and word
vectors. This repository contains code for the paper 
[_Discovering Phonesthemes with Sparse Regularization_](http://nelsonliu.me/papers/liu+levow+smith.sclem2018.pdf), 
which will be presented at the NAACL 2018 Workshop on Subword and Character Level Models.

On a high level, this code seeks to extract character-level sequences in words
that are highly predictive of a word's meaning. These submorphemic clusters of
supposed meaning are called "phonesthemes". As an example of a phonestheme, take
the word-inital cluster "gl-" --- many words that begin with "gl" have to do
with light or vision, such as "glow", "glint", "glitter", "glisten", etc. Thus,
psycholinguists have posited that "gl-" is a phonestheme.

## Installation

This project is being developed and tested
 (via [TravisCI](https://travis-ci.org/nelson-liu/phonesthemes)) on **Python
 3.6**.

[Conda](https://conda.io/) will set up a virtual environment with the exact
version of Python used for development along with all the dependencies
needed to run the code in this package.

1.  [Download and install conda](https://conda.io/docs/download.html).

2.  Create a conda environment with Python 3.6.

    ```
    conda create -n phonesthemes python=3.6
    ```

3.  Now activate the conda environment.

    ```
    source activate phonesthemes
    ```

4.  Install the required dependencies with `pip`.

    ```
    pip install -r requirements.txt
    ```

5.  Install the required SpaCy data pack.
    ```
    python -m spacy download en
    ```

## Getting Started (English)

This section details how to get started and get phonesthemes from English GloVe
vectors.

### Getting preliminary data

To download some of the data needed to run the code, run:

```
python ./scripts/data/download_data.py
```

This script download English GloVe vectors (300 dimensional, trained on 840B
tokens of Common Crawl) as well as the CMU English pronouncing dictionary.

### Filtering non-English words from the vectors

While it's possible to run the code on the raw GloVe vectors themselves, the
model will pick up on spurious form-meaning correlations. For example, it will
posit that "xv" is a phonestheme, since it appears in "xvi", "xvii", "xviii",
"xviv", etc (the roman numerals). This makes sense, since the roman numerals all
have vectors that are very similar to each other. However, it's not quite what
we're looking for.

As a result, we need to filter out non-English and rare words from our set of
vectors. To do so, we want to:

 - Discard vectors corresponding to words that are not alphabetical
 - Discard vectors corresponding to words that are not lowercase
 - Discard vectors corresponding to words that are length 2 or shorter

In addition, we perform further filtering with token frequencies from the
Gigaword corpus. We discard vectors that correspond to words occuring in more
than 50% of all Gigaword documents and those that correspond to words that occur
less than 1000 times in the corpus (these are generally misspellings, nonce
words, or rare words). If you have a copy of the Gigaword corpus and want to
produce the frequency list used here,
see
[./scripts/data/count_gigaword_tokens.py](./scripts/data/count_gigaword_tokens.py);
else, you can download the frequency list we used [here](http://nelsonliu.me/papers/phonesthemes/data/gigaword_token_total_counts.txt).

As a final filtering step, we discard words that share a lemma, but only if that
lemma has a corresponding vector. The intuition behind this is that we do not
want to weight candidate phonesthemes in frequently-occuring morphemes more
heavily. For example, if we had "glisten", "glistening", "glistened",
"glistens", we only really want to keep the uninflected form of the word
("glisten").

All of this filtering is done
by
[./scripts/data/clean_and_filter_vectors.py](./scripts/data/clean_and_filter_vectors.py).
This script takes the path to the input word vectors (in GloVe format), a path
to an input token frequencies file (which we get from Gigaword), a minimum token
count (the lowest frequency that a word must have in the frequencies file in
order to not be filtered), a maximum document ratio (any token that occurs in a
proportion of documents greater than this ratio will be thrown out) and an
output folder to write a text file with the filtered vectors to. To run, use:

```
python scripts/data/clean_and_filter_vectors.py \
    --vectors_path ./data/interim/glove.840B.300d.txt \
    --counts_path ./data/processed/gigaword_token_total_counts.txt \
    --min_token_count 1000 \
    --max_document_ratio 0.5 \
    --output_dir ./data/processed/glove.840B.300d/
```

This will output a file named `glove.840B.300d.filtered.txt` to the
`./data/processed/glove.840B.300d` directory.


### Making phoneme vectors

Phonesthemes are an inherently phonemic phenomenon, and thus it makes sense to
try to extract clusters of meaning-bearing phones (versus clusters of
meaning-bearing characters). For English, we use the CMU Pronouncing dictionary
to get phonemic (ARPABET) representations of string words.

[./scripts/data/phonemicize_vectors.py](./scripts/data/phonemicize_vectors.py) takes
a file of vectors (such as the file generated
from
[./scripts/data/clean_and_filter_vectors.py](./scripts/data/clean_and_filter_vectors.py) or
the vanilla GloVe vectors) and creates a new phonemicized vector file where each
semantic vector is associated with a sequence of phones. We discard vectors
corresponding to words that share the same phonemic representation (homophones).
To run, use:

```
python scripts/data/phonemicize_vectors.py \
    --vectors_path ./data/processed/glove.840B.300d/glove.840B.300d.filtered.txt \
    --graphemes_to_phonemes_path ./data/external/cmudict.dict \
    --output_dir ./data/processed/glove.840B.300d/
```

This will output a file of phonemicized vectors (`<comma-separated phonemes>
<space><space-delimited vector>`) at
`./data/processed/glove.840B.300d/glove.840B.300d.filtered.phonemicized.txt`
and a file of the grapheme to phoneme mapping used in the phonemicization
process (`<word><tab><space-separated phonemes>`) at
`./data/processed/glove.840B.300d/glove.840B.300d.filtered.graphemes_to_phonemes.txt`.


### Extracting Bound Morphemes

The approach above will generate form-meaning associations from the text, but it
still isn't guaranteed to give us phonesthemes since the features important to
the semantic vector of a word may be derived from morphemes. For example, the
morpheme "re-" can be mistaken as a phonestheme (it is certainly quite
predictive of a word's vector, but it isn't submorphemic).

To combat this effect, we fit an initial regression model to predict the
semantic vector from morpheme-level features. Then, we subsequently fit the
character ngram phonesthemes model on the residuals of the morpheme-level model
(`original word vectors - word vectors as predicted by the morpheme model`).
Intuitively, we are subtracting out a vector that carries meaning from
morphemes, and so we can fit the phonestheme model on the resultant residuals
and see less "morphemic" effects.

To run this model, we need a list of bound
morphemes.
[./scripts/data/get_bound_morphemes.py](./scripts/data/get_bound_morphemes.py)
extracts a list from the CELEX dataset.

```
python scripts/data/get_bound_morphemes.py \
    --celex_morph_data_path ./data/interim/celex2/english/eml/ \
    --column 22 \
    --output_dir ./data/processed/english_morphemes/
```

This script writes a list of bound morphemes to
`./data/processed/english_morphemes/eml.bound_morphemes.txt`, and it also write
a TSV of `<word>\t<space-separated morphological segmentation>` to
`./data/processed/english_morphemes/eml.seg`.

### Training the model

To train the model, use the script
at [./scripts/run/run_model.py](./scripts/run/run_model.py). This will train the
model and save it to the specified `save_dir`.

```
python scripts/run/run_model.py \
    --ngrams 2 \
    --vectors_path ./data/processed/glove.840B.300d/glove.840B.300d.filtered.phonemicized.txt \
    --mode start --graphemes_to_ phonemes_path ./data/processed/glove.840B.300d/glove.840B.300d.filtered.graphemes_to_phonemes.txt \
    --bound_morphemes_path ./data/processed/english_morphemes/eml.bound_morphemes.txt \
    --word_segmentations_path ./data/processed/english_morphemes/eml.seg \
    --min_count 5 \
    --one_hot \
    --save_dir ./models/ \
    --n_jobs 4 --run_id eng_phonemicized_bigram \
    --l1_ratio .1 .5 .7 .9 .95 .99 1
```

To load a model and output phonesthemes, use:

```
python scripts/run/run_model.py \
  --load_path models/eng_phonemicized_bigram/PhonesthemesModel_eng_phonemicized_bigram.pkl
```

## References

```
@InProceedings{liu-levow-smith:2018:SCLeM,
  author    = {Liu, Nelson F.  and  Levow, Gina-Anne  and  Smith, Noah A.},
  title     = {Discovering Phonesthemes with Sparse Regularization},
  booktitle = {Proceedings of the Second Workshop on Subword and Character Level Models in NLP},
  year      = {2018}
}
```
