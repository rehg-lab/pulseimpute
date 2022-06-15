import os


lstm_extended_ptbxl = {'modelname':'lstm', "annotate":"_extended_ptbxl", 'modeltype':'lstm', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0], "train_impute_extended":300}}

