from data.BaseDataset import BaseDataset, BaseDatasetConfig
from BaseMissingness import BaseMissingnessConfig

import os
import numpy as np
from data.BaseDataset import BaseDataset

class Config(BaseDatasetConfig):
    type = "PPGMIMIC"
    def __init__(self, path: str, missingnessconfig: BaseMissingnessConfig):
        super().__init__(path=path, missingnessconfig=missingnessconfig)

class Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def load(self, mean=False, bounds=None, train=True, val=True, test=False, **kwargs):
        missingness_config = kwargs.get('missingness', {})
        return self._process_splits(self.config.path, train, val, test, mean, bounds, missingness_config)

    def _process_splits(self, path, train, val, test, mean, bounds):
        results = []
        for split, should_load in [('train', train), ('val', val), ('test', test)]:
            if should_load:
                X = np.load(os.path.join(path, f"mimic_ppg_{split}.npy")).astype(np.float32)
                X = self.preprocess(X, mean=mean, bounds=bounds)
                X, Y_dict = self.apply_missingness(X)
                results.extend([X, Y_dict])
            else:
                results.extend([None, None])
        return tuple(results)