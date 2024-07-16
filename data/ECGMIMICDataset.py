import os
import numpy as np
from tqdm import tqdm
import torch
from csv import reader
from ast import literal_eval

from utils.missingness.mimic_missingness import MIMICMissingness

from data.PulseImputeData import PulseImputeDataset

class ECGMIMICDataset(PulseImputeDataset):

    def __init__(self):
        self.mimic_missingness = MIMICMissingness()

    def load(self, train=True, val=True, test=False, addmissing=False, path=os.path.join("data/data/mimic_ecg")):
        # note that mimic has already been mode centered and bounded 1 to -1
        if train:
            X_train = np.load(os.path.join(path, "mimic_ecg_train.npy"))
            X_train, Y_dict_train = self.applyMissingness(X_train, addmissing=addmissing, split_type="train")
        else:
            X_train = None
            Y_dict_train = None

        if val:
            X_val = np.load(os.path.join(path, "mimic_ecg_val.npy"))
            X_val, Y_dict_val = self.applyMissingness(X_val, addmissing=addmissing, split_type="val")
        else:
            X_val = None
            Y_dict_val = None

        if test:
            X_test =  np.load(os.path.join(path, "mimic_ecg_test.npy"))
            X_test, Y_dict_test = self.applyMissingness(X_test, addmissing=addmissing, split_type="test")
        else:
            X_test = None
            Y_dict_test = None

        print('SHAPES')
        print(X_test.shape)
        print(Y_dict_test['target_seq'].shape)
        return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

    def applyMissingness(self, X, addmissing=False, split_type="test"):
        return self.mimic_missingness.apply(X, data_type="ecg", split_type=split_type, addmissing=addmissing)

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
