import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .BaseDataset import BaseDataset

from utils.missingness.mimic_missingness import MIMICMissingness
from utils.missingness.extended_missingness import ExtendedMissingness
from utils.missingness.transient_missingness import TransientMissingness

class CustomDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.missingness = None

    def load(self, data_path, **kwargs):
        self.data_path = data_path
        if not self.data_path:
            raise ValueError("data_path must be specified")
        
        self.train_split = kwargs.get('train_split', 0.7)
        self.val_split = kwargs.get('val_split', 0.15)
        self.test_split = kwargs.get('test_split', 0.15)

        self.train = kwargs.get('train', True)
        self.val = kwargs.get('val', True)
        self.test = kwargs.get('test', False)
        
        if abs(self.train_split + self.val_split + self.test_split - 1) > 1e-6:
            raise ValueError("train_split, val_split, and test_split must sum to 1")
        
        self.data_load_config = kwargs
        self.missingness = self._get_missingness_instance()

        npy_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]
        
        if len(npy_files) == 1:
            data = self._load_single_file(npy_files[0])
        elif len(npy_files) == 3:
            data = self._load_multiple_files(npy_files)
        else:
            raise ValueError("Expected either 1 or 3 .npy files in the data directory")

        X_train, Y_dict_train = self._process_splits(data['train']) if self.train else (None, None)
        X_val, Y_dict_val = self._process_splits(data['val']) if self.val else (None, None)
        X_test, Y_dict_test = self._process_splits(data['test']) if self.test else (None, None)

        return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

    def _get_missingness_instance(self):
        missingness_config = self.data_load_config.get("missingness", {})
        missingness_type = missingness_config.get("missingness_type", "extended")
        if missingness_type == "extended":
            return ExtendedMissingness()
        elif missingness_type == "mimic":
            return MIMICMissingness()
        elif missingness_type == "transient":
            return TransientMissingness()
        else:
            raise ValueError(f"Unsupported missingness type: {missingness_type}")

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

    def _process_splits(self, X):
        preprocess_kwargs = {k: v for k, v in self.data_load_config.items() if k in ['Mean', 'mode', 'bounds', 'channels']}
        X = self.preprocess(X, **preprocess_kwargs)
        if len(X.shape) == 2:
            X = X[:, :, np.newaxis]
        X, Y_dict = self.apply_missingness(X)
        return X, Y_dict

    def apply_missingness(self, X):
        missingness_config = self.data_load_config.get("missingness", {})
        missingness_type = missingness_config.get("missingness_type", "extended")
        if missingness_type == "extended":
            impute_extended = missingness_config.get("impute_extended", 100)
            return self.missingness.apply(X, impute_extended)
        elif missingness_type == "mimic":
            data_type = self.data_load_config.get("data_type", "ppg")
            split_type = missingness_config.get("split_type", "train")
            addmissing = missingness_config.get("addmissing", False)
            path = missingness_config.get("path", "data/missingness_patterns")
            return self.missingness.apply(X, data_type, split_type, addmissing, path)
        elif missingness_type == "transient":
            impute_transient = missingness_config.get("impute_transient", {
                "window": 10,
                "prob": 0.5
            })
            return self.missingness.apply(X, impute_transient)
        else:
            raise ValueError(f"Unsupported missingness type: {missingness_type}")