import os
import multiprocessing
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils.loss_mask import mse_mask_loss
import scipy

from models.ClassicalModel_Architecture.ClassicalModel_Wrapper import classical


class LinearInterpolation(classical):

    def __init__(self, modelname):
        self.modelname = modelname

    def impute(self, local_batch, local_label_dict, local_batch_copy):
        local_batch_copy[~torch.isnan(local_label_dict["target_seq"])] = np.nan
        for sample in range(local_batch.shape[0]):
            imputation1d = torch.clone(local_batch_copy[sample, :, :]).squeeze(-1)
            ok = ~torch.isnan(imputation1d)
            xp = ok.ravel().nonzero().squeeze(-1)
            fp = imputation1d[~torch.isnan(imputation1d)]
            x = torch.isnan(imputation1d).ravel().nonzero().squeeze(-1)
            try:
                imputation1d[torch.isnan(imputation1d)] = torch.tensor(np.interp(x, xp, fp)).float()
            except:
                import pdb;
                pdb.set_trace()
            local_batch_copy[sample, :, :] = imputation1d.unsqueeze(-1)
        imputation = local_batch_copy
        return imputation

