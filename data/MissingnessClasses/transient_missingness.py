# utils/missingness/transient_missingness.py
import numpy as np
import torch
from .base_missingness import BaseMissingness

class TransientMissingness(BaseMissingness):
    def __init__(self, config):
        self.window = config['window']
        self.prob = config['prob']

    def apply(self, X, split):
        target = self._create_target(X)
        input = np.copy(X)
        total_len = X.shape[1]
        for i in range(X.shape[0]):
            for start_impute in range(0, total_len, self.window):
                for j in range(X.shape[-1]):
                    if np.random.random() <= self.prob:
                        end_impute = min(start_impute + self.window, total_len)
                        target[i, start_impute:end_impute, j] = X[i, start_impute:end_impute, j] 
                        input[i, start_impute:end_impute, j] = np.nan
                        X[i, start_impute:end_impute, j] = 0

        return torch.from_numpy(X), {"target_seq": torch.from_numpy(target), "input_seq": torch.from_numpy(input)}