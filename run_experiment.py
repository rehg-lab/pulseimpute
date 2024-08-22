import argparse
import yaml
from experiment.Miss_MIMICdata_Experiment import Miss_MIMIC_Experiment
from experiment.Miss_PTBXLdata_Experiment import Miss_PTBXL_Experiment
from Experiment import Experiment
from data.CustomDataset import CustomDataset
from dataclasses import dataclass
from typing import List, Optional, Dict

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
    
    # add annotate and annotate_test
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

def run_experiment(config: ExperimentConfig, bootstrap: tuple):
    validate_config(config)

    if config.data_type == 'ptbxl':
        experiment = Miss_PTBXL_Experiment(config, bootstrap)
    elif 'mimic' in config.data_type or config.data_type == 'custom':
        experiment = Miss_MIMIC_Experiment(config, bootstrap)
    else:
        raise ValueError(f"Unsupported data_type: {config.data_type}")


    if config.is_train:
        experiment.train()
    else:
        experiment.test()
        
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment with YAML config")
    parser.add_argument('-c', '--config', type=str, help="Path to YAML config file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    bootstrap = (1000, 1)  # num of bootstraps, size of bootstrap sample compared to test size

    # argparse is default, otherwise fall back to this
    config_path = 'configs/' + args.config if args.config else 'configs/FFT/fft_custom_mimic_test.yaml____'
    
    config = load_config(config_path)

    run_experiment(config, bootstrap)