from data.BaseDataset import BaseDataset
from utils.missingness.extended_missingness import ExtendedMissingness
from utils.missingness.transient_missingness import TransientMissingness

import os
import torch
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class PTBXLDataset(BaseDataset):

    def __init__(self, path="data/data/ptbxl_ecg/", seed=10):
        super().__init__()
        self.path = path
        self.seed = seed
        self.extended_missingness = ExtendedMissingness()
        self.transient_missingness = TransientMissingness()

    def load(self, mode=True, bounds=1, channels=[0], test_fold=6, val_fold=5, train_fold=4,
             train=True, val=True, test=False, **kwargs):
        missingness_config = kwargs.get('missingness', {})
        impute_extended = missingness_config.get('impute_extended')
        impute_transient = missingness_config.get('impute_transient')
        

        print("Loading PTB-XL dataset")

        if impute_extended and impute_transient:
            raise Exception('Only one missingness model should be used') #todo support multiple missingness models

        np.random.seed(self.seed)
        # load and convert annotation data
        Y = pd.read_csv(os.path.join(self.path,'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = self.load_raw_data_ptbxl(self.path)

        X_train = X[Y.strat_fold <= train_fold]
        Y_train = Y[Y.strat_fold <= train_fold]
        X_train = self.preprocess(X_train, mode=mode, bounds=bounds, channels=channels)
        X_train_processed, missingness_dict_train = self.apply_missingness(X_train, impute_extended, impute_transient)
        Y_dict_train = {"labels": Y_train, **missingness_dict_train}

        # Validation
        X_val = X[(Y.strat_fold > train_fold) & (Y.strat_fold <= val_fold)]
        Y_val = Y[(Y.strat_fold > train_fold) & (Y.strat_fold <= val_fold)]
        X_val = self.preprocess(X_val, mode=mode, bounds=bounds, channels=channels)
        X_val_processed, missingness_dict_val = self.apply_missingness(X_val, impute_extended, impute_transient)
        Y_dict_val = {"labels": Y_val, **missingness_dict_val}

        # Test
        X_test = X[Y.strat_fold > val_fold]
        Y_test = Y[Y.strat_fold > val_fold]
        X_test = self.preprocess(X_test, mode=mode, bounds=bounds, channels=channels)
        X_test_processed, missingness_dict_test = self.apply_missingness(X_test, impute_extended, impute_transient)
        Y_dict_test = {"labels": Y_test, **missingness_dict_test}
        

        return (torch.from_numpy(X_train_processed), Y_dict_train, 
                torch.from_numpy(X_val_processed), Y_dict_val, 
                torch.from_numpy(X_test_processed), Y_dict_test)

    def apply_missingness(self, X, impute_extended, impute_transient):
        if impute_extended:
            return self.extended_missingness.apply(X, impute_extended)
        elif impute_transient:
            return self.transient_missingness.apply(X, impute_transient)
        else:
            return X, {
                "target_seq": torch.from_numpy(self.extended_missingness._create_target(X)),
                "input_seq": torch.from_numpy(np.copy(X))
            }

    def load_raw_data_ptbxl(self, path):
        return np.load(os.path.join(self.path, 'ptbxl_ecg.npy'), allow_pickle=True).astype(np.float32)