from abc import ABC, abstractmethod

class PulseImputeDataset(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        """
        Load train, val, test data.
        """
        pass
        
       
    @abstractmethod
    def applyMissingness(self):
        '''
        -Returns the missingness masks
        -Returns the true values
        -Returns original with it masked
        '''
        pass

