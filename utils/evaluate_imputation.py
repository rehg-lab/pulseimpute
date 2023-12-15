import torch
import os
from .loss_mask import mse_mask_loss
import pickle
from tqdm import tqdm
from scipy.signal import find_peaks
import numpy as np
from ecgdetectors import Detectors
import pandas as pd
from tqdm.contrib.concurrent import process_map 
import itertools


"""
Functions that are used for evaluating the imputation during test time
"""


def eval_mse(imputation, target_seq, path, return_stats=False):
    mse_loss, missing_total = mse_mask_loss(torch.Tensor(imputation), target_seq, return_stats=return_stats)

    if return_stats:
        return mse_loss, missing_total
    else:
        mse_loss /= missing_total
        printlog(f"MSE: {mse_loss.item()}", path, type="w")

def eval_heartbeat_detection(imputation, target_seq, input, path, return_stats=False):
    if "ppg" in path:
        type = "ppg"
    else:
        type = "ecg"

    def get_groundtruths():
        temp_target = np.nan_to_num(target_seq) 
        nomiss_data = np.add(temp_target, input.cpu().detach().numpy())

        r_peaks_list = []
        for i in tqdm(range(nomiss_data.shape[0])):

            if type == "ppg":
                rpeaks_nomiss = np.array(find_peaks(np.array(nomiss_data[i][:,0]), prominence=.1, threshold=1e-10)[0])
            else:
                detector = Detectors(100)
                rpeaks_nomiss = np.array(detector.swt_detector(nomiss_data[i,:,0])) # to convert to ms

            r_peaks_list.append(rpeaks_nomiss)

        with open(f'data/pulseimpute_data/mimic_{type}/peaks_list.pickle', 'wb') as fp:
            pickle.dump(r_peaks_list, fp)

        return r_peaks_list
    if not os.path.exists(f'data/pulseimpute_data/mimic_{type}/peaks_list.pickle'):
        r_peaks_list = get_groundtruths()
    else:
        with open(f'data/pulseimpute_data/mimic_{type}/peaks_list.pickle', 'rb') as fp:
            r_peaks_list = pickle.load(fp)


    stats = np.array(process_map(find_peaks_all, zip(list(imputation), r_peaks_list, target_seq, itertools.repeat(type)), max_workers=os.cpu_count(), chunksize=1, total=len(r_peaks_list)))
    if return_stats:
        return stats
    else:
        sens = np.nanmean(stats[:,0])
        prec = np.nanmean(stats[:,1])
        f1 = 2*sens*prec/(sens+prec)

        printlog(f"Sensitivity: {sens}", path)
        printlog(f"Precision: {prec}", path)
        printlog(f"F1: {f1}", path)

def find_peaks_all(zipped_thing):
    imputation, true_r_peaks, target_seq, type = zipped_thing

    if type == "ppg":
        rpeaks = np.array(find_peaks(np.array(imputation[:,0]), prominence=.1, threshold=1e-10)[0])
    else:
        detector = Detectors(100)
        rpeaks = detector.swt_detector(imputation[:,0])
        rpeaks = np.array(rpeaks) # to convert to ms

    r_peaks_missing  = [] 
    for peak in true_r_peaks:
        if int(peak) in set(np.where(~torch.isnan(target_seq[:, 0]))[0]):
            r_peaks_missing.append(int(peak))

    peaks_found = []
    for peak in rpeaks:
        if int(peak) in set(np.where(~torch.isnan(target_seq[:,0]))[0]):
            peaks_found.append(int(peak))
            
    # peaks found in imputation region
    r_peaks_afterimputation_tolerable = np.concatenate((np.array(peaks_found)+2, 
                                                        np.array(peaks_found)+1, 
                                                        np.array(peaks_found), 
                                                        np.array(peaks_found)-1,
                                                        np.array(peaks_found)-2))

    # missing peaks that were not in imputed peaks
    r_peaks_thatwereNOTfoundfromoriginal = list(set(r_peaks_missing) - set(r_peaks_afterimputation_tolerable))
    true_positives = len(r_peaks_missing) - len(r_peaks_thatwereNOTfoundfromoriginal)

    # precision and sensitivity and F1
    false_negatives = len(r_peaks_thatwereNOTfoundfromoriginal)
    false_positives = len(peaks_found) - true_positives

    if true_positives + false_negatives == 0:
        return np.nan, np.nan

    sensitivity = true_positives / (true_positives + false_negatives)
    if true_positives + false_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = np.nan
    # f1 = (2*precision*sensitivity)/(precision+sensitivity)
    # f1 = true_positives / (true_positives + .5*(false_negatives + false_positives))
    return sensitivity, precision


def eval_cardiac_classification(imputation, path):

    from .ptbxl_eval_code.configs.fastai_configs import conf_fastai_xresnet1d101
    from .ptbxl_eval_code.experiments.scp_experiment import SCP_Experiment
    datafolder = 'data/pulseimpute_data/ptbxl_ecg/'
    outputfolder_pretrain = os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model")
    models = [conf_fastai_xresnet1d101]
    experiments = []
    if not os.path.exists(os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model", "trainonfoldfour_rhythm")):
        experiments.append((f'trainonfoldfour_rhythm', 'rhythm'))
    if not os.path.exists(os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model", "trainonfoldfour_form")):
        experiments.append((f'trainonfoldfour_form', 'form'))
    if not os.path.exists(os.path.join("utils", "ptbxl_eval_code", "pretrained_classification_model", "trainonfoldfour_diagnostic")):
        experiments.append((f'trainonfoldfour_diagnostic', 'diagnostic'))
    if len(experiments) > 0:
        
        for name, task in experiments:
            print(f"Training {task} Classification Model on Clean Data")
            e = SCP_Experiment(name, task, datafolder, outputfolder_pretrain, models, train_fold=4, val_fold=5, test_fold=[6,7,8,9,10])
            e.prepare(channel=0)
            e.perform()
            e.evaluate()

    print("Running classification model inference on imputation test")

    experiments = [
                    ("rhythm", 'rhythm'),
                    ("form", 'form'),
                    ("diagnostic", 'diagnostic'),
                ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, path, models, train_fold=4, val_fold=5, test_fold=[6,7,8,9,10])
        e.prepare(modelfolder=os.path.join(outputfolder_pretrain,"trainonfoldfour_"+task), data=imputation)
        e.perform(modelfolder=os.path.join(outputfolder_pretrain,"trainonfoldfour_"+task))
        e.evaluate()

    for task in ["rhythm", "form", "diagnostic" ]:
        csv_path = os.path.join(path, task, "models", "fastai_xresnet1d101", "results", "te_results.csv")
        df_temp = pd.read_csv(csv_path)
        auc = df_temp["macro_auc"][1]

        printlog(f"{task} AUC: {auc.item()}", path)

    
def printlog(line, path, type="a"):
    print(line)
    with open(os.path.join(path, 'eval_results.txt'), type) as file:
        file.write(line+'\n')

