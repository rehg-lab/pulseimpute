import os
import multiprocessing
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils.loss_mask import mse_mask_loss

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
        
        y_dict =  {"target_seq":self.imputation_dict["target_seq"][idx]}

        return X, y_dict

class lininterp():
    def __init__(self,modelname, train=True,
                train_data=None, val_data=None, data_name="", 
                 imputation_dict=None, annotate_test="",
                 annotate="", bs= 64, gpus=[0,1]):
        outpath = "out/"
            
        self.bs = bs
        self.gpu_list = gpus
        self.annotate_test = annotate_test
        self.dataname=data_name
        
        self.data_loader_setup(train_data, val_data, imputation_dict)
        
        self.ckpt_path = os.path.join(outpath, data_name+annotate_test, modelname+annotate)
        os.makedirs(self.ckpt_path, exist_ok=True)

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
        print(len(self.test_loader.dataset))

        print(f'{dt_string} | Start')

        with torch.no_grad():
            
            total_test_mse_loss = 0
            total_missing_total = 0
            flag = True
            residuals_all = []
            for local_batch, local_label_dict in tqdm(self.test_loader, desc="Testing"):
                imputation = torch.clone(local_batch) # target seq is nan at nonmask, and real valued at targets
                imputation[~torch.isnan(local_label_dict["target_seq"])] = np.nan
                for sample in range(local_batch.shape[0]):
                    imputation1d = torch.clone(imputation[sample,:,:]).squeeze(-1)
                    ok = ~torch.isnan(imputation1d)
                    xp = ok.ravel().nonzero().squeeze(-1)
                    fp = imputation1d[~torch.isnan(imputation1d)]
                    x  = torch.isnan(imputation1d).ravel().nonzero().squeeze(-1)
                    try:
                        imputation1d[torch.isnan(imputation1d)] = torch.tensor(np.interp(x, xp, fp)).float()
                    except:
                        import pdb; pdb.set_trace()
                    imputation[sample,:,:] = imputation1d.unsqueeze(-1)

                mse_loss, missing_total, residuals= mse_mask_loss(imputation, local_label_dict["target_seq"]
                                                        ,residuals=True)
                residuals_all.append(residuals)
                total_missing_total += missing_total

                total_test_mse_loss += mse_loss.item()

                if flag:
                    flag = False
                    imputation_cat = np.copy(imputation.detach().numpy())
                else:
                    imputation_temp = np.copy(imputation.detach().numpy())
                    imputation_cat = np.concatenate((imputation_cat, imputation_temp), axis=0)

        total_test_mse_loss /=  total_missing_total
        total_test_mpcl2_std = torch.std(torch.cat(residuals_all))


        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | MSE:{total_test_mse_loss:.10f} | Std SE: {total_test_mpcl2_std:.10f}  \n')
        with open(os.path.join(self.ckpt_path, "loss_log.txt"), 'a+') as f:
            f.write(f'{dt_string} | MSE:{total_test_mse_loss:.10f} | Std SE: {total_test_mpcl2_std:.10f}  \n')

        np.save(os.path.join(self.ckpt_path, "imputation.npy"), imputation_cat)
        return imputation_cat

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]