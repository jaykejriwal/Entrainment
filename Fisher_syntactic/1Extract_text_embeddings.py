import pandas as pd
import glob
import os
import sys
import torch
import numpy as np
from functools import reduce
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import stanza
import re
import math
from collections import Counter
import nltk
from nltk.util import ngrams
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
sen_w_feats = []
sentence_embeddings = []

def process_syn_text(text):         #Function to remove punctuation and extrat lemma and pos
    words = text.lower()
    doc = nlp(words)
    l=[f'{word.upos}' for sent in doc.sentences for word in sent.words]
    return ' '.join([str(elem) for elem in l])

def calculate_syntactic_similarity(text1):
    process1 = process_syn_text(text1)
    return process1

list_of_files = glob.glob('/home/jay_kejriwal/Fisher/Processed/Text/*.txt',recursive=True) 

output_path = '/home/jay_kejriwal/Fisher/Processed/Embeddings/Text_syntactic'

for file_name in list_of_files:
    out_name= os.path.join(output_path, os.path.basename(file_name))
    csv_input = pd.read_csv(file_name, usecols=[3], names=['utterance'],delimiter='\t',header=None)
    for index, row in csv_input.iterrows():
        sen_w_feats.append(row["utterance"])

    #Convert sentence to pos tag list
    sentence_vectors1 = [calculate_syntactic_similarity(i) for i in sen_w_feats] 

    vec=TfidfVectorizer()
    matrix=vec.fit_transform(sentence_vectors1)
    m=matrix.toarray()
    max_length = 17
    result = np.array([np.pad(row, (0, max_length-len(row))) for row in m])
    result1=result.tolist()

    #Merge consecutive utterance of Speaker A and B
    out = reduce(lambda x, y: x+y, result1)

    #Each consecutive utterance is of size 34 i.e 17 for each utterance
    chunks = [out[x:x+34] for x in range(0, len(out)-17, 17)]

    #Convert list to array
    arr = np.asarray(chunks)
    with open(out_name, 'w') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerows(arr)
    sen_w_feats = []
    sentence_embeddings = []
    sentence_vectors1=None
    arr=None