from abc import ABC, abstractmethod
import os

class ModelConfig():
    def __init__(self, 
                 type: str,
                 subtype: str,
                 netparams: dict):
        assert type in self.model_types_list(), "type should be a folder inside of models"
        self.type = type
        
        assert subtype in self.model_names_list(), "subtype should be a python file inside of type folder"
        self.type = subtype

        self.netparams = netparams

    def model_types_list(self):
        import pdb; pdb.set_trace()
        return [x[0] for x in os.walk(".")]
    def model_names_list(self):
        import pdb; pdb.set_trace()
        return [x[0] for x in os.walk(f"{self.modeltype}/")]


class BaseModelWrapper(ABC):
    @abstractmethod
    def data_loader_setup(self):
        """
        Set up the data loaders for the model.
        """
        pass

    #@abstractmethod
    def ckpt_load(self):
        """
        Load model checkpoints.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Fit the model on the training data.
        """
        pass

    #@abstractmethod
    def impute(self):
        """
        Perform imputation.
        """
        pass

    def train(self):
        """
        Train model.
        """
