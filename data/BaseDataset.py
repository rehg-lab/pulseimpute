from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def load(self):
        """Load train, val, test data."""
        pass

    @abstractmethod
    def apply_missingness(self, X, **kwargs):
        """Apply missingness to the data."""
        pass

    def preprocess(self, X):
        """Preprocess the data. Override this method if needed."""
        return X