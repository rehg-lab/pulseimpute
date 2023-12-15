import os
import multiprocessing
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils.loss_mask import mse_mask_loss



class nomodel():
    def __init__(self,modelname, 
                train_data=None, val_data=None, data_name="", 
                 imputation_dict=None, annotate_test="",
                 annotate="", bs= 64, gpus=[0,1]):
        self.bs = bs
        self.gpu_list = gpus
        self.annotate_test = annotate_test
        self.dataname=data_name
        outpath = "out/"
        
        if val_data is not None:
            print("Should only be used on test")
            import sys; sys.exit()
        else:
            self.test_data = train_data

        self.ckpt_path = os.path.join(outpath, data_name+annotate_test, modelname+annotate)
        print(self.ckpt_path )
        os.makedirs(self.ckpt_path, exist_ok=True)


    def testimp(self):

        np.save(os.path.join(self.ckpt_path, "imputation.npy"), self.test_data)

        return self.test_data