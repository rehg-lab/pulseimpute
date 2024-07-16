Mean_mimic_ppg_test = {'modelname':'Mean', "annotate":"_mimic_ppg",  "annotate_test":"_test",
                'modeltype':'Classical', 
                "modelparams":{},
                "data_name":"mimic_ppg","data_load": {"Mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}

Mean_mimic_ecg_test = {'modelname':'Mean', "annotate":"_mimic_ecg",  "annotate_test":"_test",
                'modeltype':'Classical', "data_name":"mimic_ecg",
                "modelparams":{},
                "data_load": {"train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}


Mean_ptbxl_testextended_10percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_10percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":100, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testextended_20percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_20percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":200, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testextended_30percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_30percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":300, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testextended_40percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_40percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":400, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testextended_50percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_50percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":500, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}

Mean_ptbxl_testtransient_10percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_10percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.10}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testtransient_20percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_20percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testtransient_30percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_30percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.30}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testtransient_40percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_40percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.40}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
Mean_ptbxl_testtransient_50percent = {'modelname':'Mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_50percent",
        'modeltype':'Classical', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.50}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}