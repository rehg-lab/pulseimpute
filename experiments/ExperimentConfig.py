from models.BaseModel import ModelConfig
from data.BaseDataset import DatasetConfig

class ExperimentConfig():
    def __init__(self, 
                 iter_save: int,
                 batch_size: int,
                 gpus: list,
                 model_config: ModelConfig,
                 dataset_config: DatasetConfig):
        
        self.iter_save = iter_save
        self.batch_size = batch_size
        self.gpus = gpus
        
        self.model_config = model_config
        self.dataset_config = dataset_config