from abc import ABC, abstractmethod
import numpy as np
import os

class BaseMissingness(ABC):
    @abstractmethod
    def apply(self, X):
        """
        Apply custom missingness to the input data.
        
        To create your own missingness pattern:
        1. Create a new file named custom_missingness.py in utils/missingness.
        2. Define a class that inherits from BaseMissingness.
        3. Implement the apply method with your custom logic.
        4. Use self._create_target(X) to initialize the target.
        5. Use self._apply_mask(X, mask) to apply the missingness mask.
        
        ------
        
        Example implementation:
        
        class CustomMissingness(BaseMissingness):
            def apply(self, X):
                target = self._create_target(X)
                mask = your_custom_mask_logic(X)
                X = self._apply_mask(X, mask)
                return X, target
        
        """
        
        raise NotImplementedError("Implement your custom missingness logic here")
    
    def _create_target(self, X):
        target = np.empty(X.shape, dtype=np.float32)
        target[:] = np.nan
        return target
    
    def _apply_mask(self, X, mask):
        X[mask] = 0
        return X
    

'''
missingness: 
    # missingness_type: extended
    # impute_extended: 100
    extended_missingness: 
      size: 100

    transient_missingness:
      window: 5
      prob: 0.1

    real_missingness:
      path: ''
'''
import pathlib
def registry(missingness_dict: dict):
    missingness_dict_possible = [name[:-3] for name in os.listdir(pathlib.Path(__file__).parent.resolve()) if name.split('.')[-1] == 'py']
    print(missingness_dict_possible)
registry({})