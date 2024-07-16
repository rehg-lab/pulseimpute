import os
import multiprocessing
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils.loss_mask import mse_mask_loss
import scipy

from models.ClassicalModel_Architecture.ClassicalModel_Wrapper import classical


class FFT(classical):

    def __init__(self, modelname):
        self.modelname = modelname

    def impute(self, local_batch, local_label_dict, local_batch_copy):
        for i in range(local_batch.shape[0]):
            startobserved = None
            endobserved = None
            startmissing = None
            endmissing = None
            endmissing_firstmissing = None
            firstmissing = False
            for j in range(local_batch.shape[1]):
                if j == 0 and ~torch.isnan(local_label_dict["target_seq"][i, j]):  # data is missing in first point
                    firstmissing = True
                    continue
                if firstmissing:  # edge case where you start with missingness
                    if not torch.isnan(local_label_dict["target_seq"][i, j]):  # nan is present, so this is when missing
                        endmissing_firstmissing = j
                        continue
                    else:
                        firstmissing = False

                if startobserved is None:  # then if we havent started tracking observed, lets start
                    startobserved = j
                if torch.isnan(local_label_dict["target_seq"][i, j]):  # we found an observed value
                    if startmissing is not None:  # if we have a missing, then find an observed
                        # we have finished the missing segment
                        endmissing = j
                else:  # we found a missing value
                    if startmissing is None:
                        startmissing = j  # we start the missing segment
                        endobserved = j  # we have finished the observed segment

                if endmissing is not None:
                    # then we begin FFT imputation
                    observed_segment = local_batch_copy[i, startobserved:endobserved]
                    fft = scipy.fft.fft(observed_segment.detach().cpu().numpy(), axis=0)

                    fft = np.concatenate((fft, np.expand_dims(np.zeros(int(endmissing - startmissing)), axis=1)))
                    if endmissing_firstmissing:
                        fft = np.concatenate((np.expand_dims(np.zeros(endmissing_firstmissing), axis=1), fft))
                    ifft = scipy.fft.ifft(fft, axis=0).real
                    if endmissing_firstmissing:
                        startobserved = 0
                        local_batch_copy[i, :endmissing_firstmissing] = torch.from_numpy(ifft[:endmissing_firstmissing])
                    local_batch_copy[i, startmissing:endmissing] = torch.from_numpy(ifft[-(endmissing - startmissing):])

                    startmissing = None
                    endmissing = None
                    endmissing_firstmissing = None
                    endobserved = None

        if not torch.allclose(local_batch_copy[torch.isnan(local_label_dict["target_seq"])],
                              local_batch[torch.isnan(local_label_dict["target_seq"])]):
            import pdb;
            pdb.set_trace()

        imputation = local_batch_copy

        return imputation

