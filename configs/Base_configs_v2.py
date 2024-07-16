from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

@dataclass
class BaseDataLoadConfig:
    dataset_name: str
    data_path: str
    splits: Dict[str, str] = field(default_factory=lambda: {"train": "train", "val": "val", "test": "test"})
    normalize: bool = False
    apply_missingness: bool = False
    missingness_params: Optional[Dict[str, Any]] = None
    custom_params: Optional[Dict[str, Any]] = None

@dataclass
class BaseModelConfig:
    model_name: str

@dataclass
class BaseTrainConfig:
    batch_size: int
    epochs: int
    optimizer: str = "adam" # adam is hard coded right now
    gpus: List[int] = field(default_factory=lambda: [0])

@dataclass
class BaseConfig:
    model_name: str # FFT
    model_type: str # Classical
    dataset_name: str # ptbxl
    experiment_name: str # 'fft_ptbxl_extended_10percent'
    mode: str  # 'train' or 'test'
    data_config: BaseDataLoadConfig
    model_config: BaseModelConfig
    train_config: BaseTrainConfig