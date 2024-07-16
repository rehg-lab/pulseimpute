import numpy as np
import torch
from csv import reader
from ast import literal_eval
import os
from .base_missingness import BaseMissingness

class MIMICMissingness(BaseMissingness):
    def apply(self, X, data_type, split_type, addmissing=False, path="data/missingness_patterns"):
        X = torch.from_numpy(X)
        if data_type == 'ecg':
            X = X.unsqueeze(-1)
        target = self._create_target(X)
        target = torch.from_numpy(target)
        
        if addmissing:
            miss_tuples_path = os.path.join(path, f"mHealth_missing_{data_type}", f"missing_{data_type}_{split_type}.csv")
            with open(miss_tuples_path, 'r') as read_obj:
                csv_reader = reader(read_obj)
                list_of_miss = list(csv_reader)
            
            for iter_idx, waveform_idx in enumerate(range(0, X.shape[0], 4)):
                miss_idx = iter_idx % len(list_of_miss)
                miss_vector = self._miss_tuple_to_vector(list_of_miss[miss_idx])
                for i in range(min(4, X.shape[0] - waveform_idx)):
                    target[waveform_idx + i, np.where(miss_vector == 0)[0]] = X[waveform_idx + i, np.where(miss_vector == 0)[0]]
                    X[waveform_idx + i, :, :] = X[waveform_idx + i, :, :] * miss_vector
        
        return X, {"target_seq": target}

    def _miss_tuple_to_vector(self, listoftuples):
        def onesorzeros_vector(miss_tuple):
            miss_tuple = literal_eval(miss_tuple)
            return np.zeros(miss_tuple[1]) if miss_tuple[0] == 0 else np.ones(miss_tuple[1])

        miss_vector = np.concatenate([onesorzeros_vector(t) for t in listoftuples])
        return np.expand_dims(miss_vector, 1)