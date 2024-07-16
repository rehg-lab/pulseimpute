from data.PulseImputeData import PulseImputeDataset

#from BaseDataset import BaseDataset
from utils.missingness.extended_missingness import ExtendedMissingness
from utils.missingness.transient_missingness import TransientMissingness

import os
import torch
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import numpy as np


class PTBXLDataset(PulseImputeDataset):

    def __init__(self, path="data/data/ptbxl_ecg/", seed=10):
        self.path = path
        self.seed = seed
        self.extended_missingness = ExtendedMissingness()
        self.transient_missingness = TransientMissingness()
        

    def load(self, mode=True, bounds=1, 
            impute_extended=None, impute_transient=None,
            channels=[0], test_fold=6, val_fold=5, train_fold=4,
            train=True, val=True, test=False # used to keep consistent with other loads
            ):

        print("Loading PTB-XL dataset")

        if impute_extended and impute_transient:
            raise Exception('Only one missingness  model should be used') #todo support multiple missingness models

        np.random.seed(self.seed)
        # load and convert annotation data
        Y = pd.read_csv(os.path.join(self.path,'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = self.load_raw_data_ptbxl(self.path) #shape samples, time, channels

        X = self.preprocess(X, mode, bounds, channels)

        X, input, target = self.applyMissingness(X, impute_extended, impute_transient)

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
    

    def preprocess(self, X, mode=True, bounds=1, channels=[0]):
            # center waveforms along mode
            if channels is not None:
                X = X[:, :, channels]
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
            if bounds is not None:
                max_val = np.amax(np.abs(X_flat), axis = 1, keepdims=True)
                X /= np.expand_dims(max_val, axis = 2)/bounds
            return X
    
    def applyMissingness(self, X, impute_extended, impute_transient):
        if impute_extended:
            return self.extended_missingness.apply(X, impute_extended)
        elif impute_transient:
            return self.transient_missingness.apply(X, impute_transient)
        else:
            return X, np.copy(X), self.extended_missingness._create_target(X)

    def load_raw_data_ptbxl(self, path):
        return np.load(os.path.join(self.path, 'ptbxl_ecg.npy'), allow_pickle=True).astype(np.float32)
