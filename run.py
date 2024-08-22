import argparse
import yaml
import os
import numpy as np
import torch
from tqdm import tqdm
from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, eval_cardiac_classification, printlog
from utils.eval.evaluate_ptbxl import evaluate_ptbxl
from utils.eval.evaluate_mimic import evaluate_mimic
from utils.random_seed import random_seed
from data.PPGMIMICDataset import PPGMIMICDataset
from data.ECGMIMICDataset import ECGMIMICDataset
from data.CustomDataset import CustomDataset
from data.PTBXLDataset import PTBXLDataset
from dataclasses import dataclass
from typing import List, Optional, Dict
from sklearn.metrics import roc_auc_score

@dataclass
class ExperimentConfig:
    modelname: str
    modeltype: str
    modelparams: Dict
    data_path: str
    data_type: str
    data_load: Dict
    is_train: bool
    train: Dict
    annotate: str = ""
    annotate_test: str = ""

def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    config_dict['annotate'] = f"_{config_dict['data_type']}"
    
    if not config_dict['is_train']:
        if config_dict['data_type'] == 'ptbxl':
            missingness_config = config_dict['data_load'].get('missingness', {})
            if missingness_config.get('missingness_type') == 'extended':
                percentage = missingness_config.get('impute_extended', 0) // 10
                config_dict['annotate_test'] = f"_testextended_{percentage}percent"
            elif missingness_config.get('missingness_type') == 'transient':
                percentage = int(missingness_config.get('impute_transient', 0).get('prob', 0) * 100)
                config_dict['annotate_test'] = f"_testtransient_{percentage}percent"
            else:
                config_dict['annotate_test'] = "_test"
        else:
            config_dict['annotate_test'] = "_test"
    
    return ExperimentConfig(**config_dict)

def validate_config(config: ExperimentConfig):
    if config.data_type not in ['mimic_ppg', 'mimic_ecg', 'ptbxl', 'custom']:
        raise ValueError(f"Invalid data_type: {config.data_type}. Supported types are 'mimic_ppg', 'mimic_ecg', 'ptbxl', and 'custom'.")
    if config.data_type == 'custom' and not config.data_path:
        raise ValueError("Data path must be specified for custom dataset.")

def get_dataset_loader(config: ExperimentConfig):
    if config.data_type == 'ptbxl':
        return PTBXLDataset()
    elif config.data_type == 'mimic_ppg':
        return PPGMIMICDataset()
    elif config.data_type == 'mimic_ecg':
        return ECGMIMICDataset()
    elif config.data_type == 'custom':
        return CustomDataset()
    else:
        raise ValueError(f"Unsupported data_type: {config.data_type}")

def train(config: ExperimentConfig):
    print(config.modelname + config.annotate)
    random_seed(10, True)
    
    dataset_loader = get_dataset_loader(config)
    X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader.load(**config.data_load, 
                                                                            train=True, val=True, test=False)
    
    model_type = config.modeltype
    model_module = __import__(f'models.{model_type}Model_Architecture.{model_type}Model_Wrapper', fromlist=[''])
    model_module_class = getattr(model_module, model_type.lower())
    model = model_module_class(modelname=config.modelname, train_data=X_train, val_data=X_val, 
                            data_type=config.data_type, annotate=config.annotate,  
                            **config.modelparams,
                            **config.train)
    model.train()

def test(config: ExperimentConfig, bootstrap: tuple):
    print(config.modelname + config.annotate + config.annotate_test)
    random_seed(10, True)
    
    dataset_loader = get_dataset_loader(config)
    
    if config.data_type == 'custom':
        load_args = {**config.data_load, 'data_path': config.data_path}
    else:
        load_args = config.data_load

    X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader.load(**load_args)


    path = os.path.join("out/out_test/", config.data_type + config.annotate_test, config.modelname + config.annotate)
    
    if os.path.exists(os.path.join(path, "imputation.npy")):
        imputation = np.load(os.path.join(path, "imputation.npy"))
    else:
        model_type = config.modeltype
        model_module = __import__(f'models.{model_type}Model_Architecture.{model_type}Model_Wrapper', fromlist=[''])
        model_module_class = getattr(model_module, model_type.lower())
        model = model_module_class(modelname=config.modelname, data_type=config.data_type, 
                                train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
                                annotate=config.annotate, annotate_test=config.annotate_test,  
                                **config.modelparams,
                                **config.train)
        imputation = model.fit()

    if not os.path.exists(os.path.join(path, "original.npy")):
        np.save(os.path.join(path, "original.npy"), X_test)
    if not os.path.exists(os.path.join(path, "target_seq.npy")):
        np.save(os.path.join(path, "target_seq.npy"), Y_dict_test['target_seq'].numpy())

    if bootstrap is not None:
        if config.data_type == 'ptbxl':
            evaluate_ptbxl(imputation, Y_dict_test, path, bootstrap)
        elif 'mimic' in config.data_type or config.data_type == 'custom':
            evaluate_mimic(imputation, Y_dict_test, X_test, path, bootstrap, config.data_type)
        else:
            raise ValueError(f"Unsupported data_type for evaluation: {config.data_type}")
    else:
        eval_mse(imputation, Y_dict_test["target_seq"], path)
        if "mimic" in config.data_type or config.data_type == 'custom':
            eval_heartbeat_detection(imputation=imputation, target_seq=Y_dict_test["target_seq"], input=X_test, path=path)
        elif config.data_type == 'ptbxl':
            eval_cardiac_classification(imputation, path)
        else:
            raise ValueError(f"Unsupported data_type for evaluation: {config.data_type}")

def run_experiment(config: ExperimentConfig, bootstrap: tuple):
    validate_config(config)

    if config.is_train:
        train(config)
    else:
        test(config, bootstrap)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment with YAML config")
    parser.add_argument('-c', '--config', type=str, help="Path to YAML config file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    bootstrap = (1000, 1)  # num of bootstraps, size of bootstrap sample compared to test size

    config_path = 'configs/' + args.config if args.config else 'configs/FFT/fft_custom_mimic_test.yaml'
    
    config = load_config(config_path)

    run_experiment(config, bootstrap)