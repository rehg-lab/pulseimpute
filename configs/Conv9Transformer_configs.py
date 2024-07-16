'''
TRAIN
'''

Conv9Transformer_mimic_ppg = {'modelname':'Conv9Transformer', "annotate":"_mimic_ppg", 'modeltype':'Transformer', 
            "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":500, "bs": 4, "gpus":[0,1,2,3], "train_realppg":True}, "is_train": True}

Conv9Transformer_mimic_ecg = {'modelname':'Conv9Transformer', "annotate":"_mimic_ecg", 'modeltype':'Transformer', 
            "data_name":"mimic_ecg","data_load": {},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "reload_epoch":"pretrain","bs": 4, "gpus":[0,1,2,3], "train_realecg":True}, "is_train": True}

Conv9Transformer_extended_ptbxl = {'modelname':'Conv9Transformer', "annotate":"_extended_ptbxl", 'modeltype':'Transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0], "train_impute_extended":300}, "is_train": True}

Conv9Transformer_packet_ptbxl = {'modelname':'Conv9Transformer', "annotate":"_packet_ptbxl", 'modeltype':'Transformer', 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "channels":[0]},
            "modelparams":{"max_len":1000},
            "train":{"bs": 64, "gpus":[0,1],  "train_impute_wind":5, "train_impute_prob":.30}, "is_train": True}



'''
TEST
'''

epoch = "best"

Conv9Transformer_mimic_ppg_test = {'modelname':'Conv9Transformer', "annotate":"_mimic_ppg", 'modeltype':'Transformer', 
             "annotate_test":"_test",
            "data_name":"mimic_ppg","data_load": {"addmissing":True, "mean":True, "bounds":1, "train":False, "val":False, "test":True},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "reload_epoch_long":60000,"bs": 4, "gpus":[0]}, "is_train": False}

Conv9Transformer_mimic_ecg_test = {'modelname':'Conv9Transformer', "annotate":"_mimic_ecg", 'modeltype':'Transformer', 
             "annotate_test":"_test",
            "data_name":"mimic_ecg","data_load": {"train":False, "val":False, "test":True, "addmissing":True},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100, "reload_epoch_long":42600,"bs": 4, "gpus":[0]}, "is_train": False}

Conv9Transformer_transient_ptbxl_testtransient_10percent = {'modelname':'Conv9Transformer', "annotate":"_transient_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testtransient_10percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.10}, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_transient_ptbxl_testtransient_20percent = {'modelname':'Conv9Transformer', "annotate":"_transient_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testtransient_20percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_transient_ptbxl_testtransient_30percent = {'modelname':'Conv9Transformer', "annotate":"_transient_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testtransient_30percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.30}, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_transient_ptbxl_testtransient_40percent = {'modelname':'Conv9Transformer', "annotate":"_transient_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testtransient_40percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.40}, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_transient_ptbxl_testtransient_50percent = {'modelname':'Conv9Transformer', "annotate":"_transient_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testtransient_50percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.50}, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}


Conv9Transformer_extended_ptbxl_testextended_10percent = {'modelname':'Conv9Transformer', "annotate":"_extended_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testextended_10percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":100, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_extended_ptbxl_testextended_20percent = {'modelname':'Conv9Transformer', "annotate":"_extended_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testextended_20percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":200, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_extended_ptbxl_testextended_30percent = {'modelname':'Conv9Transformer', "annotate":"_extended_ptbxl", 'modeltype':'Transformer',
            "annotate_test":"_testextended_30percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":300, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_extended_ptbxl_testextended_40percent = {'modelname':'Conv9Transformer', "annotate":"_extended_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testextended_40percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":400, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}
Conv9Transformer_extended_ptbxl_testextended_50percent = {'modelname':'Conv9Transformer', "annotate":"_extended_ptbxl", 'modeltype':'Transformer', 
            "annotate_test":"_testextended_50percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":500, "channels":[0]},
            "modelparams":{"reload_epoch":epoch},
            "train":{"bs": 64, "gpus":[0]}, "is_train": False}