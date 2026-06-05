import numpy as np
import torch
import pickle

with open("tt_2d_eig_mt_mean.pkl", "rb") as f:   # 'rb' = read binary
    flux = pickle.load(f)

with open("tt_2d_eig_mt_stdev.pkl", "rb") as f:   # 'rb' = read binary
    std = pickle.load(f)

print(type(flux))