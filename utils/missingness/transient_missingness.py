import numpy as np
import torch
from .base_missingness import BaseMissingness

class TransientMissingness(BaseMissingness):
    def apply(self, X, impute_transient):
        target = self._create_target(X)
        input = np.copy(X)
        total_len = X.shape[1]
        amt_impute = impute_transient["window"]
        for i in range(X.shape[0]):
            for start_impute in range(0, total_len, amt_impute):
                for j in range(X.shape[-1]):
                    rand = np.random.random_sample()
                    if rand <= impute_transient["prob"]:
                        target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j] 
                        input[i, start_impute:start_impute+amt_impute, j] = np.nan
                        X[i, start_impute:start_impute+amt_impute, j] = 0
        return X, {"target_seq": torch.from_numpy(target), "input_seq": torch.from_numpy(input)}