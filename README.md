# Entrainment using DNN

Python program for training DNN models for entrainment detection at auditory, lexical, syntactic and semantic linguistic levels.

The program for auditory entrainment detection is adapted from https://github.com/nasir0md/unsupervised-learning-entrainment

## Dataset

We utilized state-of-the-art DNN embeddings such as BERT and TRIpLet Loss network (TRILL) vectors to extract features for measuring semantic and auditory similarities of turns within dialogues in three spoken corpora, namely Columbia Games corpus, Voice Assistant conversation corpus, and Fisher corpus.


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

Two Jupyter Notebook files are provided for the CGC and VAC corpus. These files need to be executed first for feature extraction.

Firstly, LLD features can be extracted using shell script file 0feat_extract_nopre.sh
Next, the 1create_h5data.py file allows the creation of embeddings in h5 data format.
Lastly, models can be trained using different distance measures, such as L1 and cos, which are mentioned in the file.

## Citation

J. Kejriwal, Š. Beňuš and L.M. Rojas-Barahona, "Entrainment Detection Using DNN," (2024). Submitted to Computer Speech & Language Journal (under review). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4769763159

