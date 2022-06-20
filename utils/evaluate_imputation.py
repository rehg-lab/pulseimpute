import torch
import os
from .loss_mask import mse_mask_loss
import pickle
from tqdm import tqdm
from scipy.signal import find_peaks
import numpy as np
from ecgdetectors import Detectors
import random
import pandas as pd



"""
Functions that are used for evaluating the imputation during test time
"""




def eval_mse(imputation, target_seq, path):
    mse_loss, missing_total = mse_mask_loss(torch.Tensor(imputation), target_seq)
    mse_loss /= missing_total

    printlog(f"MSE: {mse_loss.item()}", path, type="w")

def eval_heartbeat_detection(imputation, target_seq, input, path):
    if "ppg" in path:
        type = "ppg"
    else:
        type = "ecg"

    def get_groundtruths():
        nomiss_data = torch.nansum(target_seq, input)

        r_peaks_list = []
        for i in tqdm(range(nomiss_data.shape[0])):

            if type == "ppg":
                rpeaks_nomiss = np.array(find_peaks(np.array(nomiss_data[i][:,0]), prominence=.1, threshold=1e-10)[0])
            else:
                detector = Detectors(100)
                rpeaks_nomiss = np.array(detector.swt_detector(nomiss_data[i]))/10 # to convert to ms

            r_peaks_list.append(rpeaks_nomiss)

        with open(f'data/mimic_{type}/peaks_list.pickle', 'wb') as fp:
            pickle.dump(r_peaks_list, fp)

    if not os.path.exists('data/mimic_{type}/peaks_list.pickle'):
        get_groundtruths()



def eval_cardiac_classification(imputation, path):

    from .ptbxl_eval_code.configs.fastai_configs import conf_fastai_xresnet1d101
    from .ptbxl_eval_code.experiments.scp_experiment import SCP_Experiment
    datafolder = 'data/data/ptbxl_ecg/'
    outputfolder_pretrain = os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model")
    models = [conf_fastai_xresnet1d101]
    experiments = []
    if not os.path.exists(os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model", "trainonfoldfour_diagnostic")):
        experiments.append((f'trainonfoldfour_diagnostic', 'diagnostic'))
    if not os.path.exists(os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model", "trainonfoldfour_form")):
        experiments.append((f'trainonfoldfour_form', 'form'))
    if not os.path.exists(os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model", "trainonfoldfour_rhythm")):
        experiments.append((f'trainonfoldfour_rhythm', 'rhythm'))
    if len(experiments) > 0:
        
        for name, task in experiments:
            print(f"Training {task} Classification Model on Clean Data")
            e = SCP_Experiment(name, task, datafolder, outputfolder_pretrain, models, train_fold=4, val_fold=5, test_fold=[6,7,8,9,10])
            e.prepare(channel=0)
            e.perform()
            e.evaluate()

    print("Running classification model inference on imputation test")

    experiments = [
                    ("diagnostic", 'diagnostic'),
                    ("form", 'form'),
                    ("rhythm", 'rhythm'),
                ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, path, models, train_fold=4, val_fold=5, test_fold=[6,7,8,9,10])
        e.prepare(modelfolder=os.path.join(outputfolder_pretrain,"trainonfoldfour_"+task), data=imputation)
        e.perform(modelfolder=os.path.join(outputfolder_pretrain,"trainonfoldfour_"+task))
        e.evaluate()

    for task in ["diagnostic", "form", "rhythm"]:
        csv_path = os.path.join(path, task, "models", "fastai_xresnet1d101", "results", "te_results.csv")
        df_temp = pd.read_csv(csv_path)
        auc = df_temp["macro_auc"][1]

        printlog(f"{task} AUC: {auc.item()}", path)

    
def printlog(line, path, type="a"):
    print(line)
    with open(os.path.join(path, 'eval_results.txt'), type) as file:
        file.write(line+'\n')

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False