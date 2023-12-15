import os
import torch
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import numpy as np


def load(path=os.path.join("data/pulseimpute_data/ptbxl_ecg/"), seed=10, 
        mode=True, bounds=1, 
        impute_extended=None, impute_transient=None,
        channels=[0], test_fold=6, val_fold=5, train_fold=4,
        train=True, val=True, test=False # used to keep consistent with other loads
        ):

    print("Loading PTB-XL dataset")

    if impute_extended and impute_transient:
        raise Exception('Only one missingness  model should be used') #todo support multiple missingness models

    np.random.seed(seed)
    # load and convert annotation data
    Y = pd.read_csv(os.path.join(path,'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(path) #shape samples, time, channels
    if channels:
        X = X[:,:,channels]

    if mode:
        # center waveforms along mode
        X_flat = X.reshape(X.shape[0], -1)
        hist_out = np.apply_along_axis(lambda a: np.histogram(a, bins=50), 1, X_flat)
        hist = hist_out[:, 0]
        bin_edges = hist_out[:, 1]
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
    if impute_extended:
        print("Adding Extended Missingness")
        total_len = X.shape[1]
        amt_impute = impute_extended
        for i in range(X.shape[0]): # iterating through all data points
            for j in range(X.shape[-1]):
                start_impute = np.random.randint(0, total_len-amt_impute)
                target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j] 
                input[i, start_impute:start_impute+amt_impute, j] = np.nan
                X[i, start_impute:start_impute+amt_impute, j] = 0

    if impute_transient:
        print("Adding Transient Missingness")
        total_len = X.shape[1]
        amt_impute = impute_transient["window"]
        for i in range(X.shape[0]):
            for start_impute in range(0, total_len, amt_impute):
                for j in range(X.shape[-1]):
                    rand = np.random.random_sample()
                    if rand <= impute_transient["prob"]:
                        target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j] 
                        input[i, start_impute:start_impute+amt_impute, j] = np.nan
                        X[i, start_impute:start_impute+amt_impute, j] = 0

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

    return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

def load_raw_data_ptbxl(path):
    data = np.load(os.path.join(path , 'ptbxl_ecg.npy'), allow_pickle=True)
    return data.astype(np.float32)
