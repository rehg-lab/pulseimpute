import os

import numpy as np
from tqdm import tqdm

import numpy as np
import torch
from csv import reader
from ast import literal_eval

import math



def load(mean=None, bounds=None, train=True, val=True, test=False, addmissing=False, path=os.path.join("data/pulseimpute_data/mimic_ppg")):

    # Train
    if train:
        X_train = np.load(os.path.join(path, "mimic_ppg_train.npy")).astype(np.float32)
        if mean:
            X_train -= np.mean(X_train,axis=1,keepdims=True) 
        if bounds:
            X_train /= np.amax(np.abs(X_train),axis=1,keepdims=True)*bounds
        X_train, Y_dict_train = modify(X_train, type="train", addmissing=False)
    else:
        X_train = None
        Y_dict_train = None

    # Val
    if val:
        X_val = np.load(os.path.join(path, "mimic_ppg_val.npy")).astype(np.float32)
        if mean:
            X_val -= np.mean(X_val,axis=1,keepdims=True) 
        if bounds:
            X_val /= np.amax(np.abs(X_val),axis=1,keepdims=True)*bounds
        X_val, Y_dict_val = modify(X_val, type="val", addmissing=False)
    else:
        X_val = None
        Y_dict_val = None

    # Test
    if test:
        X_test = np.load(os.path.join(path, "mimic_ppg_test.npy")).astype(np.float32)
        if mean:
            X_test -= np.mean(X_test,axis=1,keepdims=True) 
        if bounds:
            X_test /= np.amax(np.abs(X_test),axis=1,keepdims=True)*bounds
        X_test, Y_dict_test = modify(X_test, type="test", addmissing=addmissing)
    else:
        X_test = None
        Y_dict_test = None

    return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

def modify(X, type=None, addmissing=False):
    X = torch.from_numpy(X)
    target = np.empty(X.shape, dtype=np.float32)
    target[:] = np.nan
    target = torch.from_numpy(target)

    if addmissing:
        print("adding missing")
        miss_tuples_path = os.path.join("data","missingness_patterns", f"missing_ppg_{type}.csv")
        with open(miss_tuples_path, 'r') as read_obj:
            csv_reader = reader(read_obj)
            list_of_miss = list(csv_reader)
        # X is 15174, bf is 43
        for iter_idx, waveform_idx in enumerate(tqdm(range(0, X.shape[0], 4))):
            ppgmiss_idx = iter_idx % len(list_of_miss)
            miss_vector = miss_tuple_to_vector(list_of_miss[ppgmiss_idx])
            if X.shape[0] - waveform_idx < 4:
                totalrange = X.shape[0] - waveform_idx 
            else:
                totalrange = 4
            for i in range(totalrange): # bs of 4, for that batch, we have the same missingness pattern
                target[waveform_idx + i, np.where(miss_vector == 0)[0]] = X[waveform_idx + i, np.where(miss_vector == 0)[0]]
                X[waveform_idx + i, :, :] = X[waveform_idx + i, :, :] * miss_vector
    return X, {"target_seq":target}

def miss_tuple_to_vector(listoftuples):
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
