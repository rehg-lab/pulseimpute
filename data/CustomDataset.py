import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .BaseDataset import BaseDataset

from utils.missingness.mimic_missingness import MIMICMissingness
from utils.missingness.extended_missingness import ExtendedMissingness

class CustomDataset(BaseDataset):
    def __init__(self, data_path, preprocessor=None, missingness=None, 
                 train_split=0.7, val_split=0.15, test_split=0.15):
        self.data_path = data_path
        self.preprocessor = preprocessor or (lambda x: x)
        self.missingness = missingness or ExtendedMissingness()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def load(self, train=True, val=True, test=True, addmissing=False, impute_extended=None):
        # TEMPORARY
        self.impute_extended = True

        npy_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]
        
        if len(npy_files) == 1:
            data = self._load_single_file(npy_files[0])
        elif len(npy_files) == 3:
            data = self._load_multiple_files(npy_files)
        else:
            raise ValueError("Expected either 1 or 3 .npy files in the data directory")

        X_train, Y_dict_train = self._process_split(data['train']) if train else (None, None)
        X_val, Y_dict_val = self._process_split(data['val']) if val else (None, None)
        X_test, Y_dict_test = self._process_split(data['test']) if test else (None, None)

        return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

    def _load_multiple_files(self, npy_files):
        data = {}
        for file in npy_files:
            if 'train' in file.lower():
                data['train'] = np.load(os.path.join(self.data_path, file))
            elif 'val' in file.lower():
                data['val'] = np.load(os.path.join(self.data_path, file))
            elif 'test' in file.lower():
                data['test'] = np.load(os.path.join(self.data_path, file))
        
        if len(data) != 3:
            raise ValueError("Could not identify train, val, and test files")
        
        return data

    def _load_single_file(self, npy_file):
        data = np.load(os.path.join(self.data_path, npy_file))
        
        train_val, test = train_test_split(data, test_size=self.test_split, random_state=42)
        train, val = train_test_split(train_val, test_size=self.val_split/(self.train_split+self.val_split), random_state=42)
        
        return {'train': train, 'val': val, 'test': test}

    def _process_split(self, X):
        X = self.preprocess(X)
        #X, Y_dict = self.apply_missingness(X)
        # TEMPORARY
        if len(X.shape) == 2:
            X = X[:, :, np.newaxis]
        X, input_seq, target_seq = self.missingness.apply(X, self.impute_extended)
        X = torch.from_numpy(X).float()
        Y_dict = {"target_seq": torch.from_numpy(target_seq).float(), "input_seq": torch.from_numpy(input_seq).float()}
        
        return X, Y_dict

    def preprocess(self, X):
        return self.preprocessor(X)

    def apply_missingness(self, X, **kwargs):
        return self.missingness.apply(X, **kwargs)