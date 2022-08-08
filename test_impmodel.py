from configs.test_transformer_configs import *
from configs.test_brits_configs import *
from configs.test_other_configs import *
from configs.test_naomi_configs import *

from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, eval_cardiac_classification
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


    configs = [fft_mimic_ecg_test]


    for config in configs:
        print(config["modelname"]+config["annotate"]+config["annotate_test"])

        load = getattr(__import__(f'utils.{config["data_name"]}', fromlist=['']), "load")
        X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**config["data_load"])
        random_seed(10, True)

        path = os.path.join("out/", config["data_name"]+config["annotate_test"], config["modelname"]+config["annotate"])

        if os.path.exists(os.path.join(path, "imputation.npy")):
            imputation = np.load(os.path.join(path, "imputation.npy"))
        else:
            model_type = config["modeltype"]
            model_module = __import__(f'models.{model_type}_model', fromlist=[''])
            model_module_class = getattr(model_module, model_type)
            model = model_module_class(modelname=config["modelname"], data_name=config["data_name"], 
                                    train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
                                    annotate=config["annotate"],  annotate_test=config["annotate_test"],  
                                    **config["modelparams"],
                                    **config["train"])
            imputation = model.testimp()
   
        eval_mse(imputation, Y_dict_test["target_seq"], path)
        if "mimic" in config["data_name"]:
            eval_heartbeat_detection(imputation=imputation, target_seq=Y_dict_test["target_seq"], input=X_test, path=path)
        else:
            eval_cardiac_classification(imputation, path)



