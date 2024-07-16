from dataclasses import dataclass
from typing import List, Optional

# documentation 

@dataclass
class BaseDataLoadConfig:
    pass

@dataclass
class BaseTrainConfig:
    pass

@dataclass
class BaseModelParamsConfig:
    pass

@dataclass
class BaseConfig:
    modelname: str
    annotate: str
    annotate_test: str
    modeltype: str
    data_name: str
    is_train: bool
    data_load: BaseDataLoadConfig
    train: BaseTrainConfig
    modelparams: BaseModelParamsConfig
    data_path: Optional[str] = None