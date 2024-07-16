from abc import ABC, abstractmethod

class PulseImputeModelWrapper(ABC):
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
