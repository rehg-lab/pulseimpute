import os


conv9_emb256_layer2_mimic_ppg = {'modelname':'conv9_emb256_layer2', "annotate":"_mimic_ppg", 'modeltype':'transformer', 
            "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":500, "bs": 4, "gpus":[0,1,2,3], "train_realppg":True}}

van_emb256_posembed_layer2_mimic_ppg = {'modelname':'van_emb256_posembed_layer2', "annotate":"_mimic_ppg", 'modeltype':'transformer', 
            "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "bs": 4, "gpus":[0,1,2,3], "train_realppg":True}}

bdc883_emb256_layer2_mimic_ppg = {'modelname':'bdc883_emb256_layer2', "annotate":"_mimic_ppg", 'modeltype':'transformer', 
            "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "bs": 4, "gpus":[0,1,2,3], "train_realppg":True}}


van_emb256_posembed_layer2_mimic_ecg= {'modelname':'van_emb256_posembed_layer2', "annotate":"_mimic_ecg", 'modeltype':'transformer', 
            "data_name":"mimic_ecg","data_load": {},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "reload_epoch":"pretrain","bs": 4, "gpus":[0,1,2,3], "train_realecg":True}}
conv9_emb256_layer2_mimic_ecg = {'modelname':'conv9_emb256_layer2', "annotate":"_mimic_ecg", 'modeltype':'transformer', 
            "data_name":"mimic_ecg","data_load": {},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "reload_epoch":"pretrain","bs": 4, "gpus":[0,1,2,3], "train_realecg":True}}

bdc883_emb256_layer2_mimic_ecg= {'modelname':'bdc883_emb256_layer2', "annotate":"_mimic_ecg", 'modeltype':'transformer', 
            "data_name":"mimic_ecg","data_load": {},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "reload_epoch":"pretrain","bs": 4, "gpus":[0,1,2,3], "train_realecg":True}}


conv9_emb256_layer2_extended_ptbxl = {'modelname':'conv9_emb256_layer2', "annotate":"_extended_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0], "train_impute_extended":300}}

conv9_emb256_layer2_packet_ptbxl = {'modelname':'conv9_emb256_layer2', "annotate":"_packet_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0,1],  "train_impute_wind":5, "train_impute_prob":.30}}


bdc883_emb256_layer2_extended_ptbxl = {'modelname':'bdc883_emb256_layer2', "annotate":"_extended_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0,1,2], "train_impute_extended":300}}

bdc883_emb256_layer2_packet_ptbxl = {'modelname':'bdc883_emb256_layer2', "annotate":"_packet_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0,1,2],  "train_impute_wind":5, "train_impute_prob":.30}}


van_emb256_posembed_layer2_packet_ptbxl = {'modelname':'van_emb256_posembed_layer2', "annotate":"_packet_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1,  "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0,1], "train_impute_wind":5, "train_impute_prob":.30}}

van_emb256_posembed_layer2_extended_ptbxl = {'modelname':'van_emb256_posembed_layer2', "annotate":"_extended_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"reload_epoch":"best","bs": 64, "gpus":[0,1], "train_impute_extended":300}}



deepmvi_extended_ptbxl = {'modelname':'deepmvi', "annotate":"_extended_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0], "train_impute_extended":300}}

deepmvi_transient_ptbxl = {'modelname':'deepmvi', "annotate":"_transient_ptbxl", 'modeltype':'transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0],  "train_impute_wind":5, "train_impute_prob":.30}}

deepmvi_mimic_ecg= {'modelname':'deepmvi', "annotate":"_mimic_ecg", 'modeltype':'transformer',
        "data_name":"mimic_ecg","data_load": {},
        "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4]}},
        "train":{"iter_save":1000, "reload_epoch":"pretrain","bs": 4, "gpus":[0,1,2,3], "train_realecg":True}} 

deepmvi_mimic_ppg= {'modelname':'deepmvi', "annotate":"_mimic_ppg", 'modeltype':'transformer',
        "data_name":"mimic_ppg", "data_load": {"mean":True, "bounds":1},
        "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":1000, "bs": 4, "gpus":[0,1,2,3], "train_realppg":True}}
