from data.BaseDataset import BaseDataset, BaseDatasetConfig
from BaseMissingness import BaseMissingnessConfig

import os
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


class Config(BaseDatasetConfig):
    type = "PTBXL"
    def __init__(self, path: str, missingnessconfig: BaseMissingnessConfig):
        super().__init__(path=path, missingnessconfig=missingnessconfig)

class Dataset(BaseDataset):

    def __init__(self, config):
        super().__init__(config)
        self.seed=10

    def load(self, mode=True, bounds=1, channels=[0], test_fold=6, val_fold=5, train_fold=4,
             train=True, val=True, test=False, **kwargs):

        print("Loading PTB-XL dataset")

        np.random.seed(self.seed)
        # load and convert annotation data
        Y = pd.read_csv(os.path.join(self.config.path,'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        X = self.load_raw_data_ptbxl(self.path)

        X_train = X[Y.strat_fold <= train_fold]
        Y_train = Y[Y.strat_fold <= train_fold]
        X_train = self.preprocess(X_train, mode=mode, bounds=bounds, channels=channels)
        X_train_processed, missingness_dict_train = self.apply_missingness(X_train)
        Y_dict_train = {"labels": Y_train, **missingness_dict_train}

        X_val = X[(Y.strat_fold > train_fold) & (Y.strat_fold <= val_fold)]
        Y_val = Y[(Y.strat_fold > train_fold) & (Y.strat_fold <= val_fold)]
        X_val = self.preprocess(X_val, mode=mode, bounds=bounds, channels=channels)
        X_val_processed, missingness_dict_val = self.apply_missingness(X_val)
        Y_dict_val = {"labels": Y_val, **missingness_dict_val}

        X_test = X[Y.strat_fold > val_fold]
        Y_test = Y[Y.strat_fold > val_fold]
        X_test = self.preprocess(X_test, mode=mode, bounds=bounds, channels=channels)
        X_test_processed, missingness_dict_test = self.apply_missingness(X_test)
        Y_dict_test = {"labels": Y_test, **missingness_dict_test}

        return (X_train_processed, Y_dict_train, 
                X_val_processed, Y_dict_val, 
                X_test_processed, Y_dict_test)

    def load_raw_data_ptbxl(self, path):
        return np.load(os.path.join(self.path, 'ptbxl_ecg.npy'), allow_pickle=True).astype(np.float32)