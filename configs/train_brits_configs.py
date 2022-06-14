import os 
#### Tune BRITS PTBXL

brits_i_512_mimic_ppg =  {'modelname':'brits_i', "annotate":"_512_mimic_ppg", 'modeltype':'brits', 
            "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, 
            "brits_data_path":"data/imputation_brits", "bigfile":False},
            "data_name":"mimic_ppg",
            "data_load": {"mean":True, "bounds":1},
            "train":{"iter_save":25, "bs": 2, "gpus":[0,1], "train_realppg":True, "createjson":True}}

brits_i_512_mimic_ecg =  {'modelname':'brits_i', "annotate":"_512_mimic_ecg", 'modeltype':'brits', 
            "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, 
            "brits_data_path":"data/imputation_brits", "bigfile":False},
            "data_name":"mimic_ecg",
            "data_load": {"bf":False, "path":"data"},
            "train":{"iter_save":25, "reload_epoch":"pretrain","bs": 2, "gpus":[0,1], "train_realecg":True, "createjson":True}}

brits_i_512_transient_ptbxl= {'modelname':'brits_i', "annotate":"_512_transient_ptbxl", 'modeltype':'brits', 
            "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0},
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "train":{"train_time":60*60*24,"bs": 32, "gpus":[0], "train_impute_wind":5, "train_impute_prob":.30}}

brits_i_512_extended_ptbxl= {'modelname':'brits_i', "annotate":"_512_extended_ptbxl", 'modeltype':'brits', 
            "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0},
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "train":{"train_time":60*60*24,"bs": 32, "gpus":[0], "train_impute_extended":300}}