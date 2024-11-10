import os
import numpy as np
from .BaseDataset import BaseDataset
from sklearn.model_selection import train_test_split

class CustomDataset(BaseDataset):
    def __init__(self):
        super().__init__()

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
        missingness_config = kwargs.get('missingness', {})

        npy_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]

        if len(npy_files) == 1:
            data = self._load_single_file(npy_files[0])
        elif len(npy_files) == 3:
            data = self._load_multiple_files(npy_files)
        else:
            raise ValueError("Expected either 1 or 3 .npy files in the data directory")

        X_train, Y_dict_train = self._process_splits(data['train'], 'train', missingness_config) if self.train else (None, None)
        X_val, Y_dict_val = self._process_splits(data['val'], 'val', missingness_config) if self.val else (None, None)
        X_test, Y_dict_test = self._process_splits(data['test'], 'test', missingness_config) if self.test else (None, None)

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

    def _process_splits(self, X, split, missingness_config):
        preprocess_kwargs = {k: v for k, v in self.data_load_config.items() if k in ['Mean', 'mode', 'bounds', 'channels']}
        X = self.preprocess(X, **preprocess_kwargs)
        if len(X.shape) == 2:
            X = X[:, :, np.newaxis]
        X, Y_dict = self.apply_missingness(X, {**missingness_config, 'split': split})
        return X, Y_dict