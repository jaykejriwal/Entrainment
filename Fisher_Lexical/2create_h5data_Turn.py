import csv
import h5py
import numpy as np
import pandas as pd
import glob
import random
import pdb

SEED=448
frac_train = 1.0



# Create h5 files


sessList= sorted(glob.glob('/home/jay_kejriwal/Fisher/Processed/Embeddings/Text_lexical/*.txt',recursive=True))

num_files_all = len(sessList)
num_files_train = int(np.ceil((frac_train*num_files_all)))

sessTrain = sessList[:num_files_train]

# Create Train Data file

X_train =np.array([])
X_train = np.empty(shape=(0, 0), dtype='float64' )
for sess_file in sessTrain:
    df_i = pd.read_csv(sess_file)
    xx=np.array(df_i)
    X_train=np.vstack([X_train, xx]) if X_train.size else xx


X_train = X_train.astype('float64')
hf = h5py.File(r'/home/jay_kejriwal/Fisher/Processed/h5/lexical/train_nonorm.h5', 'w')
hf.create_dataset('textdataset', data=X_train)
hf.close()


