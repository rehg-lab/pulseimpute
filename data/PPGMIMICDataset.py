import os

from utils.missingness.mimic_missingness import MIMICMissingness

from data.PulseImputeData import PulseImputeDataset
import numpy as np
from tqdm import tqdm

import numpy as np
import torch
from csv import reader
from ast import literal_eval

import math

class PPGMIMICDataset(PulseImputeDataset):

    def __init__(self):
        self.mimic_missingness = MIMICMissingness()

    def load(self, Mean=None, bounds=None, train=True, val=True, test=False, addmissing=False, path=os.path.join("data/data/mimic_ppg")):

        # Train
        if train:
            X_train = np.load(os.path.join(path, "mimic_ppg_train.npy")).astype(np.float32)
            if Mean:
                X_train -= np.mean(X_train,axis=1,keepdims=True) 
            if bounds:
                X_train /= np.amax(np.abs(X_train),axis=1,keepdims=True)*bounds
            X_train, Y_dict_train = self.applyMissingness(X_train, split_type="train", addmissing=False)
        else:
            X_train = None
            Y_dict_train = None

        # Val
        if val:
            X_val = np.load(os.path.join(path, "mimic_ppg_val.npy")).astype(np.float32)
            if Mean:
                X_val -= np.mean(X_val,axis=1,keepdims=True) 
            if bounds:
                X_val /= np.amax(np.abs(X_val),axis=1,keepdims=True)*bounds
            X_val, Y_dict_val = self.applyMissingness(X_val, split_type="val", addmissing=False)
        else:
            X_val = None
            Y_dict_val = None

        # Test
        if test:
            X_test = np.load(os.path.join(path, "mimic_ppg_test.npy")).astype(np.float32)
            if Mean:
                X_test -= np.mean(X_test,axis=1,keepdims=True) 
            if bounds:
                X_test /= np.amax(np.abs(X_test),axis=1,keepdims=True)*bounds
            print(X_test.shape)
            X_test, Y_dict_test = self.applyMissingness(X_test, split_type="test", addmissing=addmissing)
        else:
            X_test = None
            Y_dict_test = None

        return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

    def applyMissingness(self, X, addmissing=False, split_type="train"):
        return self.mimic_missingness.apply(X, data_type="ppg", split_type=split_type, addmissing=addmissing)

    def miss_tuple_to_vector(self, listoftuples):
        def onesorzeros_vector(miss_tuple):
            miss_tuple = literal_eval(miss_tuple)
            if miss_tuple[0] == 0:
                return np.zeros(miss_tuple[1])
            elif miss_tuple[0] == 1:
                return np.ones(miss_tuple[1])

        miss_vector = onesorzeros_vector(listoftuples[0])
        for i in range(1, len(listoftuples)):
            miss_vector = np.concatenate((miss_vector, onesorzeros_vector(listoftuples[i])))
        miss_vector = np.expand_dims(miss_vector, 1)
        return miss_vector
