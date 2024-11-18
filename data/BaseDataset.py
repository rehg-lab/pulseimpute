import numpy as np
# from utils.missingness.registry import apply_missingness
from utils.utils import BaseConfig
from BaseMissingness import BaseMissingnessConfig
import os


class BaseDatasetConfig(BaseConfig):
    type = None
    def __init__(self, 
                 path: str,
                 missingnessconfig: BaseMissingnessConfig,
                ):
        assert self.type is not None
        
        self.path = path
        self.missingnessconfig = missingnessconfig


class BaseDataset:
    def __init__(self, config: BaseDatasetConfig):
        """Initialize the dataset."""

        self.config = config

        miss_module_name = config.missingnessconfig.config_fileorigin()[:-3].replace("/", ".")
        miss_module = __import__(miss_module_name, fromlist=[''])
        self.miss_class = getattr(miss_module, "Missingness")(config.missingnessconfig)

    def load(self, **kwargs):
        """Load train, val, test data."""
        raise NotImplementedError("Subclasses must implement this method")

    def apply_missingness(self, X):
        return self.miss_class.apply(X)

    def preprocess(self, X, mean=False, mode=False, bounds=None, channels=None):
        """
        Preprocess the data with options for centering and normalization.
        
        Args:
        X (np.array): Input data of shape (samples, time, channels)
        mean (bool): center data by subtracting mean
        mode (bool): center data by subtracting mode
        bounds (float): normalize data to [-bounds, bounds]
        channels (list): List of channel indices to keep. If None, keep all channels.
        
        """
        if channels is not None:
            X = X[:, :, channels]
        
        if mean:
            X -= np.mean(X, axis=1, keepdims=True)

        if mode:
            # TEMP -> for loop
            X_flat = X.reshape(X.shape[0], -1)

            hist_out = [np.histogram(a, bins=50) for a in X_flat]
            modes = np.array([np.mean([bin_edges[np.argmax(h)], bin_edges[np.argmax(h) + 1]]) 
                            for h, bin_edges in hist_out])

            X -= np.expand_dims(modes, axis=(1, 2))

            
        
        if bounds is not None:
            max_val = np.amax(np.abs(X.reshape(X.shape[0], -1)), axis=1, keepdims=True)
            X /= np.expand_dims(max_val, axis=2) / bounds
        
        return X