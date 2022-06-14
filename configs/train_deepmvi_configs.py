import os

deepmvi_extended_ptbxl = {'modelname':'deepmvi', "annotate":"_extended_ptbxl", 'modeltype':'deepmvi', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[6], "train_impute_extended":300}}

deepmvi_transient_ptbxl = {'modelname':'deepmvi', "annotate":"_transient_ptbxl", 'modeltype':'deepmvi', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[7],  "train_impute_wind":5, "train_impute_prob":.30}}

deepmvi_mimic_ecg= {'modelname':'deepmvi', "annotate":"_mimic_ecg", 'modeltype':'deepmvi',
        "data_name":"mimic_ecg","data_load": {},
        "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
        "train":{"iter_save":1000, "reload_epoch":"pretrain","bs": 4, "gpus":[0,1,2,3], "train_realecg":True}} 

deepmvi_mimic_ppg= {'modelname':'deepmvi', "annotate":"_mimic_ppg", 'modeltype':'deepmvi',
        "data_name":"mimic_ppg", "data_load": {"mean":True, "bounds":1},
        "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":1000, "bs": 4, "gpus":[0,1,2,3], "train_realppg":True}}
