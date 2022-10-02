class missingness_config():
    def __init__(self,
                 miss_transient=None,
                 miss_extended=None,
                 miss_realppg=None,
                 miss_realecg=None):

        assert bool(miss_transient) ^ bool(miss_extended) ^ bool(
            miss_realppg) ^ bool(miss_realecg)

        if miss_transient:
            self.miss_type = "miss_transient"
            self.miss = miss_transient
        elif miss_extended:
            self.miss_type = "miss_extended"
            self.miss = miss_extended
        elif miss_realppg:
            self.miss_type = "miss_realppg"
        elif miss_realecg:
            self.miss_type = "miss_realecg"


tutorial_extended_ptbxl = {"modelname": "tutorialv1",
                           "modeltype": 'tutorial',
                           "annotate": "_extended_ptbxl",
                           "data_name": "ptbxl",
                           "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                           "modelparams": {},
                           "train": {"bs": 128, "gpus": [0],
                                     "missingness_config": missingness_config(miss_extended=300)
                                     }
                           }

tutorial_transient_ptbxl = {"modelname": "tutorialv1",
                            "modeltype": 'tutorial',
                            "annotate": "_transient_ptbxl",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "train": {"bs": 128, "gpus": [0],
                                      "missingness_config": missingness_config(miss_transient={"wind": 5, "prob": .3})
                                      }
                            }
