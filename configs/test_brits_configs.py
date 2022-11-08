import os

            
brits_i_512_mimic_ppg_test = {'modelname':'brits_i', "annotate":"_512_mimic_ppg", 'modeltype':'brits', 
            "annotate_test":"_test",
            "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, 
            "brits_data_path":"/data/imputation_brits", "bigfile":False},
            "data_name":"mimic_ppg",
            "data_load": {"addmissing":True, "mean":True, "bounds":1, "train":False, "val":False, "test":True},
            "train":{"bs": 96, "gpus":[0,1,2], "createjson":True}}
            
brits_i_512_mimic_ecg_test= {'modelname':'brits_i', "annotate":"_512_mimic_ecg", 'modeltype':'brits', 
            "annotate_test":"_test",
            "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, 
            "brits_data_path":"/data/imputation_brits", "bigfile":False},
            "data_name":"mimic_ecg",
            "data_load": {"train":False, "val":False, "test":True, "addmissing":True},
            "train":{"bs": 96, "gpus":[0,1,2], "createjson":True}}    


brits_i_512_extended_ptbxl_testextended_10percent= {'modelname':'brits_i', "annotate":"_512_extended_ptbxl", 
                     "annotate_test":"_testextended_10percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1, "impute_extended":100,
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_extended_ptbxl_testextended_20percent= {'modelname':'brits_i', "annotate":"_512_extended_ptbxl", 
                     "annotate_test":"_testextended_20percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1, "impute_extended":200,
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_extended_ptbxl_testextended_30percent= {'modelname':'brits_i', "annotate":"_512_extended_ptbxl", 
                     "annotate_test":"_testextended_30percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1, "impute_extended":300,
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_extended_ptbxl_testextended_40percent= {'modelname':'brits_i', "annotate":"_512_extended_ptbxl", 
                     "annotate_test":"_testextended_40percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1, "impute_extended":400,
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_extended_ptbxl_testextended_50percent= {'modelname':'brits_i', "annotate":"_512_extended_ptbxl", 
                     "annotate_test":"_testextended_50percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1, "impute_extended":500,
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}

brits_i_512_transient_ptbxl_testtransient_10percent= {'modelname':'brits_i', "annotate":"_512_transient_ptbxl", 
                     "annotate_test":"_testtransient_10percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1, "impute_transient":{"window":5, "prob":.10},
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_transient_ptbxl_testtransient_20percent= {'modelname':'brits_i', "annotate":"_512_transient_ptbxl", 
                     "annotate_test":"_testtransient_20percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1, "impute_transient":{"window":5, "prob":.20},
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_transient_ptbxl_testtransient_30percent= {'modelname':'brits_i', "annotate":"_512_transient_ptbxl", 
                     "annotate_test":"_testtransient_30percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1,"impute_transient":{"window":5, "prob":.30},
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_transient_ptbxl_testtransient_40percent= {'modelname':'brits_i', "annotate":"_512_transient_ptbxl", 
                     "annotate_test":"_testtransient_40percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1,"impute_transient":{"window":5, "prob":.40},
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}
brits_i_512_transient_ptbxl_testtransient_50percent= {'modelname':'brits_i', "annotate":"_512_transient_ptbxl", 
                     "annotate_test":"_testtransient_50percent", 'modeltype':'brits', 
                     "data_name":"ptbxl",
                    "modelparams":{"rnn_hid_size":512, "impute_weight":1, "label_weight":0, "reload_epoch":"best", "bigfile":False},
                    "data_load": {"mode":True, "bounds":1,"impute_transient":{"window":5, "prob":.50},
                    "channels":[0]},
                    "train":{"bs": 64, "gpus":[0]}}






