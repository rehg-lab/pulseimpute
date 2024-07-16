import yaml
from configs.BDCTransformer_configs import *
from configs.DeepMVITransformer_configs import *
from configs.VanillaTransformer_configs import *
from configs.Conv9Transformer_configs import *
from configs.FFT_configs import *
from configs.Mean_configs import *
from configs.LinearInterpolation_configs import *
from configs.NAOMI_configs import *
from configs.BRITS_configs import *

from configs.FFT_new_configs import all_fft_configs

from experiment.Miss_MIMICdata_Experiment import Miss_MIMIC_Experiment
from experiment.Miss_PTBXLdata_Experiment import Miss_PTBXL_Experiment
from data.CustomDataset import CustomDataset

all_configs = {**all_fft_configs}

def validate_configs(config_list):
    # Check for more than one training config..
    train_configs = [config for config in config_list if config.is_train]
    if len(train_configs) > 1:
        raise ValueError("There can be only one training configuration.")

def categorize_configs(config_list):
    ptbxl_test = []
    mimic_test = []
    ptbxl_train = None
    mimic_train = None

    for config in config_list:
        if config.data_name == 'ptbxl':
            if config.is_train:
                ptbxl_train = config
            else:
                ptbxl_test.append(config)
        elif 'mimic' in config.data_name or config.data_name == 'custom':
            if config.is_train:
                mimic_train = config
            else:
                mimic_test.append(config)

    return ptbxl_test, mimic_test, ptbxl_train, mimic_train

def get_config(user_config):
    model = user_config['model']
    task = user_config['task']
    mode = user_config['mode']
    percentage = user_config.get('percentage')

    config_key = f"{model}_{task}_{mode}"
    if task == 'ptbxl' and mode == 'test':
        config_key += f"extended_{percentage}percent"
    elif task == 'custom':
        config_key = f"{model}_custom_{mode}"

    if config_key not in all_fft_configs:
        raise ValueError(f"Unsupported configuration: model={model}, task={task}, mode={mode}")

    config = all_fft_configs[config_key]
    
    # Add data_path to the config if it's a custom dataset
    if task == 'custom':
        config.data_path = user_config['data_path']
    
    return config


def validate_user_config(user_config):
    model = user_config['model']
    task = user_config['task']
    mode = user_config['mode']

    if model not in ['FFT', 'BDC']:
        raise ValueError(f"Invalid model: {model}. Supported models are 'FFT' and 'BDC'.")
    if task not in ['mimic_ppg', 'mimic_ecg', 'ptbxl', 'custom']:
        raise ValueError(f"Invalid task: {task}. Supported tasks are 'mimic_ppg', 'mimic_ecg', 'ptbxl', and 'custom'.")
    if mode not in ['train', 'test']:
        raise ValueError(f"Invalid mode: {mode}. Supported modes are 'train' and 'test'.")
    if task == 'ptbxl' and mode == 'test' and 'percentage' in user_config:
        percentage = user_config['percentage']
        if percentage not in [10, 20, 30, 40, 50]:
            raise ValueError(f"Invalid percentage: {percentage}. Supported percentages for PTBXL test are 10, 20, 30, 40, and 50.")
    if task == 'custom' and 'data_path' not in user_config:
        raise ValueError("Data path must be specified for custom dataset.")

def run_experiment(config_list, bootstrap):
    validate_configs(config_list)
    ptbxl_test, mimic_test, ptbxl_train, mimic_train = categorize_configs(config_list)

    print(ptbxl_test, mimic_test)

    # run experiment
    if ptbxl_train:
        experiment = Miss_PTBXL_Experiment(ptbxl_train, bootstrap)
        experiment.train()
    if mimic_train:
        experiment = Miss_MIMIC_Experiment(mimic_train, bootstrap)
        experiment.train()
    if ptbxl_test:
        experiment = Miss_PTBXL_Experiment(ptbxl_test, bootstrap)
        experiment.test()
    if mimic_test:
        experiment = Miss_MIMIC_Experiment(mimic_test, bootstrap)
        experiment.test()

if __name__ == '__main__':
    bootstrap = (1000, 1)  # num of bootstraps, size of bootstrap sample compared to test size

    with open('config.yaml', 'r') as file:
        user_config = yaml.safe_load(file)

    validate_user_config(user_config)

    config = get_config(user_config)

    run_experiment([config], bootstrap)