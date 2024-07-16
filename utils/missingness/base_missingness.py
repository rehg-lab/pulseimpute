from abc import ABC, abstractmethod
import numpy as np

class BaseMissingness(ABC):
    @abstractmethod
    def apply(self, X):
        # Your custom missingness logic here
        # Use self._create_target(X) to initialize the target
        # Use self._apply_mask(X, mask) to apply the missingness mask
        
        # Example:
        # target = self._create_target(X)
        # mask = your_custom_mask_logic(X, **kwargs)
        # X = self._apply_mask(X, mask)
        # return X, target
        
        raise NotImplementedError("Implement your custom missingness logic here")
    
    def _create_target(self, X):
        target = np.empty(X.shape, dtype=np.float32)
        target[:] = np.nan
        return target
    
    def _apply_mask(self, X, mask):
        X[mask] = 0
        return X