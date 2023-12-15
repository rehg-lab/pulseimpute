import os
import numpy as np
from tqdm import tqdm
import torch
from csv import reader
from ast import literal_eval


def load(train=True, val=True, test=False, addmissing=False, path=os.path.join("data/pulseimpute_data/mimic_ecg")):
    # note that mimic has already been mode centered and bounded 1 to -1
    if train:
        X_train = np.load(os.path.join(path, "mimic_ecg_train.npy"))
        X_train, Y_dict_train = modify(X_train, addmissing=addmissing, type="train")
    else:
        X_train = None
        Y_dict_train = None

    if val:
        X_val = np.load(os.path.join(path, "mimic_ecg_val.npy"))
        X_val, Y_dict_val = modify(X_val, addmissing=addmissing, type="val")
    else:
        X_val = None
        Y_dict_val = None

    if test:
        X_test =  np.load(os.path.join(path, "mimic_ecg_test.npy"))
        X_test, Y_dict_test = modify(X_test, addmissing=addmissing, type="test")
    else:
        X_test = None
        Y_dict_test = None

    return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

def modify(X, addmissing=False, type=None):
    X = torch.from_numpy(X)
    X = X.unsqueeze(-1)

    target = np.empty(X.shape, dtype=np.float32)
    target[:] = np.nan
    target = torch.from_numpy(target)

    if addmissing: # this is only activated during testing
        miss_tuples_path = os.path.join("data","missingness_patterns", f"missing_ecg_{type}.csv")
        with open(miss_tuples_path, 'r') as read_obj:
            csv_reader = reader(read_obj)
            list_of_miss = list(csv_reader)

        # X is 44096 missing is 10220
        for iter_idx, waveform_idx in enumerate(tqdm(range(0, X.shape[0], 4))):
            for i in range(4): 
                miss_idx = iter_idx % len(list_of_miss)
                miss_vector = miss_tuple_to_vector(list_of_miss[miss_idx])
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
