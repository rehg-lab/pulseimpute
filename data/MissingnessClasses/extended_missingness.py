import numpy as np
import torch
from data.BaseMissingness import BaseMissingness
from data.BaseMissingness import BaseMissingnessConfig


class Config(BaseMissingnessConfig):
    type = "EXTENDED"

    def __init__(self, length: int):
        self.length = length


class Missingness(BaseMissingness):
    def __init__(self, config):
        self.config = config

    def apply(self, X):
        target = self._create_target(X)
        input = np.copy(X)
        total_len = X.shape[1]
        amt_impute = self.length
        for i in range(X.shape[0]):
            for j in range(X.shape[-1]):
                start_impute = np.random.randint(0, total_len-amt_impute)
                target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j] 
                input[i, start_impute:start_impute+amt_impute, j] = np.nan
                X[i, start_impute:start_impute+amt_impute, j] = 0

        return torch.from_numpy(X), {"target_seq": torch.from_numpy(target), "input_seq": torch.from_numpy(input)}