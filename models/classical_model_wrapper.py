import os
import multiprocessing
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils.loss_mask import mse_mask_loss
import scipy
class generic_dataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, imputation_dict):
        'Initialization'
        self.waveforms = waveforms
        self.imputation_dict = imputation_dict
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.waveforms)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Load data and get label
        X = torch.clone(self.waveforms[idx, :, :])

        y_dict =  {"target_seq" :self.imputation_dict["target_seq"][idx]}

        return X, y_dict


class ClassicalModelWrapper:
    def __init__(self, modelname,
                 train_data=None, val_data=None, data_name="",
                 imputation_dict=None, annotate_test="",
                 annotate="", bs=64, gpus=[0, 1]):
        self.bs = bs
        self.gpu_list = gpus
        self.annotate_test = annotate_test
        self.dataname = data_name
        outpath = "out/"

        self.data_loader_setup(train_data, val_data, imputation_dict)

        # cannot get relative import working for the life of me

        self.ckpt_path = os.path.join(outpath, data_name + annotate_test, modelname + annotate)
        print(self.ckpt_path)
        os.makedirs(self.ckpt_path, exist_ok=True)

    def impute(self, local_batch, local_label_dict, local_batch_copy):
        raise NotImplementedError("Subclasses must implement this method")

    def data_loader_setup(self, train_data, val_data=None, imputation_dict=None):
        # num_threads_used = multiprocessing.cpu_count() // 8* len(self.gpu_list)
        num_threads_used = multiprocessing.cpu_count()
        print(f"Num Threads Used: {num_threads_used}")
        os.environ["MP_NUM_THREADS"]=str(num_threads_used)
        os.environ["OPENBLAS_NUM_THREADS"]=str(num_threads_used)
        os.environ["MKL_NUM_THREADS"]=str(num_threads_used)
        os.environ["VECLIB_MAXIMUM_THREADS"]=str(num_threads_used)
        os.environ["NUMEXPR_NUM_THREADS"]=str(num_threads_used)

        if val_data is not None:
            print("no training needed")
            import sys; sys.exit()
        else:
            if True or self.dataname != "mimic":
                test_dataset = generic_dataset(waveforms=train_data, imputation_dict=imputation_dict)
                self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
            else:
                self.test_loader = torch.utils.data.DataLoader(train_data, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)

    def testimp(self):
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")

        print(f'{dt_string} | Start')
        print(len(self.test_loader.dataset))

        with torch.no_grad():

            total_test_mse_loss = 0
            total_missing_total = 0
            flag = True
            residuals_all = []
            for local_batch, local_label_dict in tqdm(self.test_loader, desc="Testing", leave=False):
                local_batch_copy = torch.clone(local_batch)

                imputation = self.impute(local_batch, local_label_dict, local_batch_copy)

                mse_loss, missing_total, residuals = mse_mask_loss(imputation, local_label_dict["target_seq"]
                                                                   , residuals=True)
                residuals_all.append(residuals)
                total_missing_total += missing_total

                total_test_mse_loss += mse_loss.item()
                if flag:
                    flag = False
                    imputation_cat = np.copy(imputation.detach().numpy())
                else:
                    imputation_temp = np.copy(imputation.detach().numpy())
                    imputation_cat = np.concatenate((imputation_cat, imputation_temp), axis=0)

        total_test_mse_loss /= total_missing_total
        total_test_mpcl2_std = torch.std(torch.cat(residuals_all))

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | MSE:{total_test_mse_loss:.10f} | Std SE: {total_test_mpcl2_std:.10f}  \n')
        with open(os.path.join(self.ckpt_path, "loss_log.txt"), 'a+') as f:
            f.write(f'{dt_string} | MSE:{total_test_mse_loss:.10f} | Std SE: {total_test_mpcl2_std:.10f}  \n')

        np.save(os.path.join(self.ckpt_path, "imputation.npy"), imputation_cat)

        return imputation_cat