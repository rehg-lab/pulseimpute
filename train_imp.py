from configs.train_brits_configs import *
from configs.train_transformer_configs import *
from configs.train_naomi_configs import *
from configs.train_tutorial_configs import *




if __name__=='__main__':

    config = tutorial_extended_ptbxl #naomi_britsgail_mimic_ecg 

    print(config["modelname"]+config["annotate"])

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



