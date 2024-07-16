import os
import multiprocessing
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils.loss_mask import mse_mask_loss
import scipy

from models.ClassicalModel_Architecture.ClassicalModel_Wrapper import classical


class Mean(classical):

    def __init__(self, modelname):
        self.modelname = modelname

    def impute(self, local_batch, local_label_dict, local_batch_copy):
        local_batch_copy[~torch.isnan(local_label_dict["target_seq"])] = np.nan
        means = np.nanmean(local_batch_copy, axis=1)
        imputation = torch.zeros(local_batch.shape) + np.expand_dims(means, axis=1)
        imputation[torch.isnan(local_label_dict["target_seq"])] = 0
        imputation = imputation + local_batch

        return imputation

