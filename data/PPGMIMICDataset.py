import os
from utils.missingness.mimic_missingness import MIMICMissingness
from utils.missingness.extended_missingness import ExtendedMissingness
from utils.missingness.transient_missingness import TransientMissingness
from data.BaseDataset import BaseDataset
import numpy as np
import torch
from ast import literal_eval

class PPGMIMICDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.missingness_handlers = {
            'mimic': MIMICMissingness(),
            'extended': ExtendedMissingness(),
            'transient': TransientMissingness()
        }

    def load(self, Mean=False, bounds=None, train=True, val=True, test=False, **kwargs):
        missingness_config = kwargs.get('missingness', {})
        path = os.path.join("data/data/mimic_ppg")
        return self._process_splits(path, train, val, test, Mean, bounds, missingness_config)

    def _process_splits(self, path, train, val, test, Mean, bounds, missingness_config):
        results = []
        for split, should_load in [('train', train), ('val', val), ('test', test)]:
            if should_load:
                X = np.load(os.path.join(path, f"mimic_ppg_{split}.npy")).astype(np.float32)
                X = self.preprocess(X, Mean=Mean, bounds=bounds)
                X, Y_dict = self.apply_missingness(X, split, missingness_config)
                results.extend([X, Y_dict])
            else:
                results.extend([None, None])
        return tuple(results)

    def apply_missingness(self, X, split_type, missingness_config):
        missingness_type = missingness_config.get('missingness_type', 'mimic')
        missingness_handler = self.missingness_handlers.get(missingness_type)
        
        if missingness_handler is None:
            raise ValueError(f"Unsupported missingness type: {missingness_type}")
        
        if missingness_type == 'mimic':
            return missingness_handler.apply(X, data_type="ppg", split_type=split_type, 
                                             addmissing=missingness_config.get('addmissing', False))
        elif missingness_type == 'extended':
            return missingness_handler.apply(X, missingness_config.get('impute_extended', 100))
        elif missingness_type == 'transient':
            return missingness_handler.apply(X, missingness_config.get('impute_transient', {'window': 10, 'prob': 0.5}))