import numpy as np
import torch
from .base_missingness import BaseMissingness

class ExtendedMissingness(BaseMissingness):
    def __init__(self, config):
        self.impute_extended = config['impute_extended']

    def apply(self, X, split):
        target = self._create_target(X)
        input = np.copy(X)
        total_len = X.shape[1]
        amt_impute = self.impute_extended
        for i in range(X.shape[0]):
            for j in range(X.shape[-1]):
                start_impute = np.random.randint(0, total_len-amt_impute)
                target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j] 
                input[i, start_impute:start_impute+amt_impute, j] = np.nan
                X[i, start_impute:start_impute+amt_impute, j] = 0

        return torch.from_numpy(X), {"target_seq": torch.from_numpy(target), "input_seq": torch.from_numpy(input)}