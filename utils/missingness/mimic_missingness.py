import os
import numpy as np
import torch
from csv import reader
from ast import literal_eval
from .base_missingness import BaseMissingness

class MimicMissingness(BaseMissingness):
    def __init__(self, config):
        self.path = config['path']

    def apply(self, X, split):
        X = torch.from_numpy(X)
        if len(X.shape) == 2:
            X = X.unsqueeze(-1)
        target = self._create_target(X)
        target = torch.from_numpy(target)

        csv_files = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        miss_tuples_path = next((f for f in csv_files if split in f), None)
        
        if not miss_tuples_path:
            raise ValueError(f"No CSV file found for split: {split}")

        miss_tuples_path = os.path.join(self.path, miss_tuples_path)
        
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