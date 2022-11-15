from configs.train_brits_configs import *
from configs.train_transformer_configs import *
from configs.train_naomi_configs import *
from configs.train_tutorial_configs import *


import numpy as np
import torch
import random


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

if __name__=='__main__':

    config = tutorial_transient_ptbxl 

    print(config["modelname"]+config["annotate"])
    random_seed(10, True)
    load = getattr(__import__(f'utils.{config["data_name"]}', fromlist=['']), "load")
    X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**config["data_load"], 
                                                                            train=True, val=True, test=False)
    model_type = config["modeltype"]
    model_module = __import__(f'models.{model_type}_model', fromlist=[''])
    model_module_class = getattr(model_module, model_type)
    model = model_module_class(modelname=config["modelname"], train_data=X_train, val_data=X_val, 
                               data_name=config["data_name"], annotate=config["annotate"],  
                               **config["modelparams"],
                               **config["train"])

    model.train()

