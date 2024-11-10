import os
import sys
import re
import glob
import pickle
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
import ast
import h5py

import multiprocessing

import numpy as np


# DATA PROCESSING STUFF

def load(path=os.path.join("../../data"), seed=10, sampling_rate=100, mode=True, bounds=1, 
        impute_chunk=None, impute_packet=None,
        impute_allchannelsimultaneously=False, skip=False, torch=True, channels=[0],
        train=True,val=True,test=False):
    print("loading dataset")
    if skip:
        return None, None, None, None, None, None
    if torch:
        import torch
    else:
        import tensorflow as tf

    if impute_chunk and impute_packet:
        print("pick one imputation strategy dude")
        import sys; sys.exit()

    np.random.seed(seed)
    # load and convert annotation data
    Y = pd.read_csv(os.path.join(path,'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path) #shape samples, time, channels
    if channels:
        X = X[:,:,channels]

    if mode:
        X_flat = X.reshape(X.shape[0], -1)

        import warnings
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
        hist_out = np.apply_along_axis(lambda a: np.histogram(a, bins=50), 1, X_flat) # means we are applying function on this variable
        hist = hist_out[:, 0]
        bin_edges = hist_out[:, 1]
        # for i in range(X_flat.shape[1]):
        #     hist, bin_edges = np.histogram(X_flat[i], bins=50)
        
        def find_mode(hist, bin_edges):

            max_idx = np.argwhere(hist == np.max(hist))[0]
            mode = np.mean([bin_edges[max_idx], bin_edges[1+max_idx]])
            
            return mode
            
        modes = np.vectorize(find_mode)(hist, bin_edges)
        X -= np.expand_dims(modes, axis = (1,2))
        if bounds:
            max_val = np.amax(np.abs(X_flat), axis = 1, keepdims=True)
            X /= np.expand_dims(max_val, axis = 2)/bounds

    target = np.empty(X.shape, dtype=np.float32)
    target[:] = np.nan
    input = np.copy(X)
    if impute_chunk:
        total_len = X.shape[1]
        amt_impute = impute_chunk

        for i in range(X.shape[0]): # iterating through all data points
            if impute_allchannelsimultaneously:
                start_impute = np.random.randint(0, total_len-amt_impute)
                target[i, start_impute:start_impute+amt_impute, :] = X[i, start_impute:start_impute+amt_impute, :] 
                input[i, start_impute:start_impute+amt_impute, :] = np.nan
                X[i, start_impute:start_impute+amt_impute, :] = 0
            else:
                for j in range(12):
                    start_impute = np.random.randint(0, total_len-amt_impute)
                    target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j] 
                    input[i, start_impute:start_impute+amt_impute, j] = np.nan
                    X[i, start_impute:start_impute+amt_impute, j] = 0

    if impute_packet:
        total_len = X.shape[1]
        amt_impute = impute_packet["window"]

        for i in range(X.shape[0]):
            for start_impute in range(0, total_len, amt_impute):
                if impute_allchannelsimultaneously:
                    rand = np.random.random_sample()
                    if rand <= impute_packet["prob"]:
                        target[i, start_impute:start_impute+amt_impute, :] = X[i, start_impute:start_impute+amt_impute, :] 
                        input[i, start_impute:start_impute+amt_impute, :] = np.nan
                        X[i, start_impute:start_impute+amt_impute, :] = 0
                else:
                    for j in range(12):
                        rand = np.random.random_sample()
                        if rand <= impute_packet["prob"]:
                            target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j] 
                            input[i, start_impute:start_impute+amt_impute, j] = np.nan
                            X[i, start_impute:start_impute+amt_impute, j] = 0

    # test_fold = 5
    # val_fold = 4
    # train_fold=3
    test_fold = 6
    val_fold = 5
    train_fold=4
    if torch:
        # Train
        X_train = torch.from_numpy(X[Y.strat_fold <= train_fold])
        Y_dict_train = {"labels":Y[Y.strat_fold <= train_fold],
                        "target_seq":torch.from_numpy(target[Y.strat_fold <= train_fold]),
                        "input_seq":torch.from_numpy(input[Y.strat_fold <= train_fold])}
        # Val
        X_val = torch.from_numpy(X[Y.strat_fold == val_fold])
        Y_dict_val = {"labels":Y[Y.strat_fold == val_fold],
                        "target_seq":torch.from_numpy(target[Y.strat_fold == val_fold]),
                        "input_seq":torch.from_numpy(input[Y.strat_fold == val_fold])}
        # Test
        X_test = torch.from_numpy(X[Y.strat_fold >= test_fold])
        Y_dict_test = {"labels":Y[Y.strat_fold >= test_fold],
                        "target_seq":torch.from_numpy(target[Y.strat_fold >= test_fold]),
                        "input_seq":torch.from_numpy(input[Y.strat_fold >= test_fold])}
    else:
        # Train
        X_train = tf.convert_to_tensor(X[Y.strat_fold <= train_fold])
        Y_dict_train = {"labels":Y[Y.strat_fold <= train_fold]}
        # Val
        X_val = tf.convert_to_tensor.from_numpy(X[Y.strat_fold == val_fold])
        Y_dict_val = {"labels":Y[Y.strat_fold == val_fold]}
        # Test
        X_test = tf.convert_to_tensor.from_numpy(X[Y.strat_fold == test_fold])
        Y_dict_test = {"labels":Y[Y.strat_fold == test_fold]}


    return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test
    
def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        h5f = h5py.File(os.path.join(path, 'raw100.h5'),'r')
        data = np.array(h5f['data'][:])
        h5f.close()
    elif sampling_rate == 500:
        data = np.load(os.path.join(path, 'raw500.npy'), allow_pickle=True)
    return data.astype(np.float32)

# def load_raw_data_ptbxl(df, sampling_rate, path):
#     if sampling_rate == 100:
#         if os.path.exists(os.path.join(path , 'raw100.npy')):
#             data = np.load(os.path.join(path , 'raw100.npy'), allow_pickle=True)
#         else:
#             data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df.filename_lr)]
#             data = np.array([signal for signal, meta in data])
#             pickle.dump(data, open(os.path.join(path , 'raw100.npy'), 'wb'), protocol=4)
#     elif sampling_rate == 500:
#         if os.path.exists(os.path.join(path, 'raw500.npy')):
#             data = np.load(os.path.join(path, 'raw500.npy'), allow_pickle=True)
#         else:
#             data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df.filename_hr)]
#             data = np.array([signal for signal, meta in data])
#             pickle.dump(data, open(os.path.join(path, 'raw500.npy'), 'wb'), protocol=4)
#     return data.astype(np.float32)
