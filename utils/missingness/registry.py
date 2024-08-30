import os
import importlib
from pathlib import Path

def get_missingness_class(missingness_config):
    missingness_type = next(iter(missingness_config))
    missingness_dict_possible = [name[:-3] for name in os.listdir(Path(__file__).parent) if name.endswith('.py') and name != 'registry.py']
    
    if missingness_type not in missingness_dict_possible:
        raise ValueError(f"Unsupported missingness type: {missingness_type}")
    
    module = importlib.import_module(f'.{missingness_type}', package='utils.missingness')
    class_name = ''.join(word.capitalize() for word in missingness_type.split('_'))
    
    if not hasattr(module, class_name):
        raise AttributeError(f"Module {module.__name__} has no attribute {class_name}")
    
    missingness_class = getattr(module, class_name)
    return missingness_class(missingness_config[missingness_type])

def apply_missingness(X, missingness_config):
    missingness_class = get_missingness_class(missingness_config)
    split = missingness_config.get('split')
    if split is None:
        raise ValueError("Split must be specified in the missingness configuration")
    return missingness_class.apply(X, split)