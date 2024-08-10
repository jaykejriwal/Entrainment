# Entrainment using DNN

Python program for training DNN models for entrainment detection at auditory, lexical, syntactic and semantic linguistic levels.

The program for auditory entrainment detection is adapted from https://github.com/nasir0md/unsupervised-learning-entrainment

## Dataset

We utilized state-of-the-art DNN embeddings such as BERT and TRIpLet Loss network (TRILL) vectors to extract features for measuring lexical, syntactic, semantic and auditory similarities of turns within dialogues in three spoken corpora, namely Columbia Games corpus, Voice Assistant conversation corpus, and Fisher corpus.


## Required Software

ffmpeg (Download from https://www.ffmpeg.org/download.html)

sph2pipe (Download from https://www.openslr.org/3/)

opensmile (https://github.com/audeering/opensmile)

sentence-transformers (pip install sentence-transformers)

tensorflow (pip install tensorflow)

textgrid (Install textgrid from https://github.com/kylebgorman/textgrid)

TRILL vectors model (Download from https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3)

## Execution instruction

The programs need to be executed in a sequential format. 

Two Jupyter Notebook files are provided for the CGC and VAC corpora. These files need to be executed first to extract features from the respective corpora.

Each folder needs to be executed in sequential format for training DNN models on a specific linguistic level

For instance,
For training DNN models on LLD features
Fisher_acoustic/LLD
Scripts should be executed in the following manner

First, the shell script needs to be executed 0feat_extract_nopre.sh (This script will extract LLD embeddings from the Fisher corpus)

Second, 1create_h5data.py (This script will create h5 files from embeddings)

Third, 2train_testwith1random_cos.py (The script will train models with cosine distance as distance measure with one random variable)

Fourth, 2train_testwith1random_l1.py (The script will train models with l1 distance as distance measure with one random variable)

Fifth, 2train_testwith10random.py (The script will train models and compare distance with the mean of ten random variables)

Sixth, baseline.py (The script will measure baseline accuracy with l1 distance without training the DNN models)

Lastly, baselinewithcos.py (The script will measure baseline accuracy with cosine distance without training the DNN models)

Similarly, Fisher_lexical will allow us to extract lexical features and train the models.

## Citation

J. Kejriwal, Š. Beňuš and L.M. Rojas-Barahona, "Entrainment Detection Using DNN," (2024). Submitted to Computer Speech & Language Journal (under review). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4769763159

