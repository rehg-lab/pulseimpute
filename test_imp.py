from models.test_deepattn_configs import *
from models.test_brits_configs import *
from models.test_other_configs import *
from models.test_naomi_configs import *
from models.test_deepmvi_configs import * 

import argparse

parser = argparse.ArgumentParser(description='What to train on')
parser.add_argument('--name', metavar='n', type=str,  default="no config given",
                    help='Name of Config')




if __name__=='__main__':
    args = parser.parse_args()
    print(args.name)

    configs = [deepmvi_extendedptbxlpretrain_win800dil4_mimictest]


    for config in configs:
        print(config["annotate_test"]+config["modelname"]+config["annotate"])

        load = getattr(__import__(f'utils.{config["data_name"]}', fromlist=['']), "load")
        X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**config["data_load"])

        # # create json object from dictionary
        # if not os.path.exists(os.path.join("out", config["data_name"] + "_labels.csv")): # readdddddddddd
        #     Y_dict_test["labels"].to_csv(os.path.join("out", config["data_name"] + "_labels.csv"))

        model_type = config["modeltype"]
        model_module = __import__(f'models.{model_type}_model', fromlist=[''])
        model_module_class = getattr(model_module, model_type)
        model = model_module_class(modelname=config["modelname"], data_name=config["data_name"], 
                                train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
                                annotate=config["annotate"],  annotate_test=config["annotate_test"],  
                                **config["modelparams"],
                                **config["train"])
        model.testimp()

# num2words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', \
#              6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', \
#             11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', \
#             15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', \
#             19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty', \
#             50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', \
#             90: 'ninety', 0: 'zero'}
# def n2w(n):
#     try:
#         return num2words[n]
#     except KeyError:
#         try:
#             return num2words[n-n%10] + num2words[n%10].lower()
#         except KeyError:
#             print('Number out of range')


# if __name__=='__main__':
#     args = parser.parse_args()
#     print(args.name)

#     configs = [naomi_step256_ppg_test]

#     i = 15

#     config = configs[0]
#     config["data_load"]["amt"] = (i/16, (i+1)/16)
#     config["modelparams"]["save_batches"] =  n2w(i) + config["modelparams"]["save_batches"]

#     print(config["annotate_test"]+config["modelname"]+config["annotate"])

#     load = getattr(__import__(f'utils.{config["data_name"]}', fromlist=['']), "load")
#     X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**config["data_load"])

#     # # create json object from dictionary
#     # if not os.path.exists(os.path.join("out", config["data_name"] + "_labels.csv")): # readdddddddddd
#     #     Y_dict_test["labels"].to_csv(os.path.join("out", config["data_name"] + "_labels.csv"))

#     model_type = config["modeltype"]
#     model_module = __import__(f'models.{model_type}_model', fromlist=[''])
#     model_module_class = getattr(model_module, model_type)
#     model = model_module_class(modelname=config["modelname"], data_name=config["data_name"], 
#                             train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
#                             annotate=config["annotate"],  annotate_test=config["annotate_test"],  
#                             **config["modelparams"],
#                             **config["train"])
#     model.testimp()





# import os
# import multiprocessing
# import torch
# import numpy as np
# from datetime import datetime
# from tqdm import tqdm
# from utils.loss_mask import mse_mask_loss
# from utils.viz import make_impplot_12chan

# from fancyimpute import KNN
# from tqdm.contrib.concurrent import process_map 
# from npy_append_array import NpyAppendArray

# import warnings
# warnings.filterwarnings("ignore")

# class generic_dataset(torch.utils.data.Dataset):
#     def __init__(self, waveforms, imputation_dict):
#         'Initialization'
#         self.waveforms = waveforms
#         self.imputation_dict = imputation_dict
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.waveforms)
#     def __iter__(self):
#         self.idx = 0
#         return self

#     def __next__(self):
#         X = torch.clone(self.waveforms[self.idx, :, :])
#         y_dict =  {"target_seq":self.imputation_dict["target_seq"][self.idx]}

#         self.idx += 1
#         if self.idx < self.waveforms.shape[0]:
#             return X, y_dict
#         else:
#             raise StopIteration
        
    
# class knn():
#     def __init__(self,modelname, 
#                 train_data=None, val_data=None, data_name="", 
#                  imputation_dict=None, annotate_test="",
#                  annotate="", bs= 64, gpus=[0,1]):
#         self.bs = bs
#         self.gpu_list = gpus
#         self.annotate_test = annotate_test
#         self.dataname=data_name
        
#         self.data_loader_setup(train_data, val_data, imputation_dict)
        
#         self.ckpt_path = os.path.join("out", data_name+annotate_test, modelname+annotate)
#         os.makedirs(self.ckpt_path, exist_ok=True)


#     def data_loader_setup(self, train_data, val_data=None, imputation_dict=None):
#         # num_threads_used = multiprocessing.cpu_count() // 8* len(self.gpu_list)
#         num_threads_used = 1
#         print(f"Num Threads Used: {num_threads_used}")

#         if val_data is not None:
#             print("no training needed")
#             import sys; sys.exit()
#         else:
#             if self.dataname != "mimic":
#                 self.test_dataset = generic_dataset(waveforms=train_data, imputation_dict=imputation_dict)
#             else:
#                 self.test_loader = torch.utils.data.DataLoader(train_data, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)

    
#     def testimp(self):
#         dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
#         self.KNN_imputer_class = KNN(k=10, verbose=False)
#         self.epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(0))
#         os.makedirs(self.epoch_check_path, exist_ok=True)
#         self.npaa = NpyAppendArray(os.path.join(self.epoch_check_path, "imputation.npy"))

#         if os.path.exists(os.path.join(self.epoch_check_path, "mse_loss_components.txt")):
#             os.remove(os.path.join(self.epoch_check_path, "mse_loss_components.txt"))

#         print(f'{dt_string} | Start')
#         with torch.no_grad():
            
#             self.total_test_mse_loss = 0
#             flag = True
            
#             self.imputation_cat = []

#             print("trying to do process map")
#             process_map(self.operation, self.test_dataset, max_workers=os.cpu_count(), chunksize=1) 

#         with open(os.path.join(self.epoch_check_path, "mse_loss_components.txt"), 'r') as fin:
#             total_test_mse_loss = sum(int(line) for line in fin if line.strip().isnumeric())

#         total_test_mse_loss /=  len(self.test_dataset)

#         dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
#         with open(os.path.join(self.epoch_check_path, "loss_log.txt"), 'a+') as f:
#             f.write(f'{dt_string} | MSE:{total_test_mse_loss:.10f} \n')



#     def operation(self, local_tuple):
#         local_batch,local_label_dict = local_tuple
#         imputation = torch.clone(local_batch)
#         imputation[torch.isnan(local_label_dict["target_seq"])] = np.nan
#         imputation = torch.from_numpy(self.KNN_imputer_class.fit_transform(imputation.squeeze(0))).unsqueeze(0)

#         mse_loss = mse_mask_loss(imputation, local_label_dict["target_seq"].unsqueeze(0))
#         with open(os.path.join(self.epoch_check_path, "mse_loss_components.txt"), 'a+') as f:
#             f.write(str(mse_loss.item()) + "\n")

#         if self.dataname != "mimic":
#             self.npaa.append(imputation.cpu().detach().numpy())
