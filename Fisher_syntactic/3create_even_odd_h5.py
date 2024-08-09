import h5py
import numpy as np
import pandas as pd


path= r'/home/jay_kejriwal/Fisher/Processed/h5/syntactic/train_nonorm.h5'
new_path = r'/home/jay_kejriwal/Fisher/Processed/h5/syntactic/train_nonorm_even.h5'
new_path1 = r'/home/jay_kejriwal/Fisher/Processed/h5/syntactic/train_nonorm_odd.h5'

with h5py.File(path, 'r') as f:
   data_set = f['textdataset']
   new_data_even = data_set[::2]
   new_data_odd = data_set[1::2]

with h5py.File(new_path, 'w') as f:
   f.create_dataset('textdataset', data=new_data_even)

with h5py.File(new_path1, 'w') as f:
   f.create_dataset('textdataset', data=new_data_odd)