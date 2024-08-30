import os
import numpy as np
import torch
from data.BaseDataset import BaseDataset

class ECGMIMICDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def load(self, train=True, val=True, test=False, **kwargs):
        missingness_config = kwargs.get('missingness', {})
        path = os.path.join("data/data/mimic_ecg")
        return self._process_splits(path, train, val, test, missingness_config)

    def _process_splits(self, path, train, val, test, missingness_config):
        results = []
        for split, should_load in [('train', train), ('val', val), ('test', test)]:
            if should_load:
                X = np.load(os.path.join(path, f"mimic_ecg_{split}.npy"))
                X = self.preprocess(X, **missingness_config.get('preprocess', {}))
                X, Y_dict = self.apply_missingness(X, {**missingness_config, 'split': split})
                results.extend([X, Y_dict])
            else:
                results.extend([None, None])
        return tuple(results)