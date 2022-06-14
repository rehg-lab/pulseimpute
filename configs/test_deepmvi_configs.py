deepmvi_mimic_ecg_test = {'modelname':'deepmvi', "annotate":"_mimic_ecg", 'modeltype':'deepmvi', 
             "annotate_test":"_test",
            "data_name":"mimic_ecg","data_load": {"train":False, "val":False, "test":True, "addmissing":True},
            "modelparams":{"reload_epoch_long":"1892000","convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":100,"bs": 4, "gpus":[0], "train_realecg":True}}

            
deepmvi_ppg_test = {'modelname':'deepmvi', "annotate":"_mimic_ppg", 'modeltype':'deepmvi', 
             "annotate_test":"_test",
            "data_name":"mimic_ppg","data_load": {"addmissing":True, "mean":True, "bounds":1, "train":False, "val":False, "test":True},
            "modelparams":{"reload_epoch_long":"1202000", "convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":1000, "bs": 4, "gpus":[0,1,2,3], "train_realppg":True}}


deepmvi_transient_ptbxl_testtransient_10percent = {'modelname':'deepmvi', "annotate":"_transient_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testtransient_10percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.10}, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_transient_ptbxl_testtransient_20percent = {'modelname':'deepmvi', "annotate":"_transient_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testtransient_20percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_transient_ptbxl_testtransient_30percent = {'modelname':'deepmvi', "annotate":"_transient_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testtransient_30percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.30}, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_transient_ptbxl_testtransient_40percent = {'modelname':'deepmvi', "annotate":"_transient_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testtransient_40percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.40}, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_transient_ptbxl_testtransient_50percent = {'modelname':'deepmvi', "annotate":"_transient_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testtransient_50percent", 
            "data_name":"ptbxl", "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.50}, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}



deepmvi_extended_ptbxl_testextended_10percent = {'modelname':'deepmvi', "annotate":"_extended_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testextended_10percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":100, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_extended_ptbxl_testextended_20percent = {'modelname':'deepmvi', "annotate":"_extended_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testextended_20percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":200, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_extended_ptbxl_testextended_40percent = {'modelname':'deepmvi', "annotate":"_extended_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testextended_40percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":400, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_extended_ptbxl_testextended_50percent = {'modelname':'deepmvi', "annotate":"_extended_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testextended_50percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":500, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}
deepmvi_extended_ptbxl_testextended_30percent = {'modelname':'deepmvi', "annotate":"_extended_ptbxl", 'modeltype':'deepmvi', 
            "annotate_test":"_testextended_30percent", 
            "data_name":"ptbxl","data_load": {"mode":True, "bounds":1, "impute_extended":300, "channels":[0]},
            "modelparams":{"reload_epoch":"best"},
            "train":{"bs": 64, "gpus":[0,1]}}



