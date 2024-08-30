import numpy as np
from utils.missingness.registry import apply_missingness

class BaseDataset:
    def __init__(self, **kwargs):
        """Initialize the dataset."""
        pass

    def load(self, **kwargs):
        """Load train, val, test data."""
        raise NotImplementedError("Subclasses must implement this method")

    def apply_missingness(self, X, missingness_config):
        return apply_missingness(X, missingness_config)

    def preprocess(self, X, Mean=False, mode=False, bounds=None, channels=None):
        """
        Preprocess the data with options for centering and normalization.
        
        Args:
        X (np.array): Input data of shape (samples, time, channels)
        Mean (bool): center data by subtracting mean
        mode (bool): center data by subtracting mode
        bounds (float): normalize data to [-bounds, bounds]
        channels (list): List of channel indices to keep. If None, keep all channels.
        
        """
        if channels is not None:
            X = X[:, :, channels]
        
        if Mean:
            X -= np.mean(X, axis=1, keepdims=True)
        
        if mode:
            X_flat = X.reshape(X.shape[0], -1)
            hist_out = np.apply_along_axis(lambda a: np.histogram(a, bins=50), 1, X_flat)
            modes = np.array([np.mean([bin_edges[np.argmax(h)], bin_edges[np.argmax(h)+1]]) 
                              for h, bin_edges in zip(hist_out[:, 0], hist_out[:, 1])])
            X -= np.expand_dims(modes, axis=(1,2))
        
        if bounds is not None:
            max_val = np.amax(np.abs(X.reshape(X.shape[0], -1)), axis=1, keepdims=True)
            X /= np.expand_dims(max_val, axis=2) / bounds
        
        return X