# from data.MissingnessClasses.extended_missingness import Config
import os
import glob

MISSINGNESS_REGISTRY = {}
for file in glob.glob("data/MissingnessClasses/*.py"):
    module_name = file[:-3].replace("/", ".")

    model_module = __import__(module_name, fromlist=[''])
    ### kevin_todo, make sure all things are called Config and Missingness classes in each
    model_module_class = getattr(model_module, "Config") 

    MISSINGNESS_REGISTRY[model_module_class.type] = model_module_class


DATASET_REGISTRY = {}
for file in glob.glob("data/DatasetClasses/*.py"):
    module_name = file[:-3].replace("/", ".")

    model_module = __import__(module_name, fromlist=[''])
    ### kevin_todo, make sure all things are called Config and Dataset classes in each
    model_module_class = getattr(model_module, "Config") 

    DATASET_REGISTRY[model_module_class.type] = model_module_class


import pdb; pdb.set_trace()