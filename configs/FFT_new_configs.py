from dataclasses import dataclass
from typing import List, Optional, Dict
from configs.Base_configs import BaseDataLoadConfig, BaseTrainConfig, BaseModelParamsConfig, BaseConfig

all_fft_configs = {}

# move data load to base_configs
@dataclass
class FFTDataLoadConfig(BaseDataLoadConfig):
    Mean: Optional[bool] = None
    bounds: Optional[int] = None
    train: Optional[bool] = None
    val: Optional[bool] = None
    test: Optional[bool] = None
    addmissing: Optional[bool] = None
    mode: Optional[bool] = None
    impute_extended: Optional[int] = None
    impute_transient: Optional[Dict[str, float]] = None
    channels: Optional[List[int]] = None

@dataclass
class FFTTrainConfig(BaseTrainConfig):
    bs: int
    gpus: List[int]

@dataclass
class FFTModelParamsConfig(BaseModelParamsConfig):
    pass


all_fft_configs["FFT_mimic_ppg_test"] = BaseConfig(
    modelname='FFT',
    annotate="_mimic_ppg",
    annotate_test="_test",
    modeltype='Classical',
    data_name="mimic_ppg",
    is_train=False,
    data_load=FFTDataLoadConfig(Mean=True, bounds=1, train=False, val=False, test=True, addmissing=True),
    train=FFTTrainConfig(bs=512, gpus=[0]),
    modelparams=FFTModelParamsConfig()
)

all_fft_configs["FFT_mimic_ecg_test"] = BaseConfig(
    modelname='FFT',
    annotate="_mimic_ecg",
    annotate_test="_test",
    modeltype='Classical',
    data_name="mimic_ecg",
    is_train=False,
    data_load=FFTDataLoadConfig(train=False, val=False, test=True, addmissing=True),
    train=FFTTrainConfig(bs=512, gpus=[0]),
    modelparams=FFTModelParamsConfig()
)

all_fft_configs["FFT_custom_test"] = BaseConfig(
    modelname='FFT',
    annotate="_custom",
    annotate_test="_test",
    modeltype='Classical',
    data_name="custom",
    is_train=False,
    data_load=FFTDataLoadConfig(train=False, val=False, test=True, addmissing=True),
    train=FFTTrainConfig(bs=512, gpus=[0]),
    modelparams=FFTModelParamsConfig()
)

for percent in [10, 20, 30, 40, 50]:
    all_fft_configs[f"FFT_ptbxl_testextended_{percent}percent"] = BaseConfig(
        modelname='FFT',
        annotate="_ptbxl",
        annotate_test=f"_testextended_{percent}percent",
        modeltype='Classical',
        data_name="ptbxl",
        is_train=False,
        data_load=FFTDataLoadConfig(mode=True, bounds=1, impute_extended=percent*10, channels=[0]),
        train=FFTTrainConfig(bs=128, gpus=[0]),
        modelparams=FFTModelParamsConfig()
    )


for percent in [10, 20, 30, 40, 50]:
    all_fft_configs[f"FFT_ptbxl_testtransient_{percent}percent"] = BaseConfig(
        modelname='FFT',
        annotate="_ptbxl",
        annotate_test=f"_testtransient_{percent}percent",
        modeltype='Classical',
        data_name="ptbxl",
        is_train=False,
        data_load=FFTDataLoadConfig(mode=True, bounds=1, impute_transient={"window": 5, "prob": percent/100}, channels=[0]),
        train=FFTTrainConfig(bs=128, gpus=[0]),
        modelparams=FFTModelParamsConfig()
    )