from abc import ABC, abstractmethod

class PulseImputeExperiment(ABC):
    
    @abstractmethod
    def __init__(self, configs, bootstrap):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

