import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.evaluate_imputation import eval_mse, eval_cardiac_classification, printlog

def evaluate_ptbxl(imputation, Y_dict_test, path, bootstrap):
    mse_losses, missing_totals = eval_mse(imputation, Y_dict_test["target_seq"], path, return_stats=True)
    printlog(f"MSE: {(torch.sum(mse_losses)/torch.sum(missing_totals)).item()}", path, type="w")
    mse_losses_bootstraplist = []

    eval_cardiac_classification(imputation, path)
    stats_true = {}
    stats_pred = {}
    for category in ["rhythm", "form", "diagnostic"]:
        y_test_true = np.load(os.path.join(path, category, "data", "y_test.npy"), allow_pickle=True)
        stats_true[category] = y_test_true
        y_test_pred = np.load(os.path.join(path, category, "models", "fastai_xresnet1d101", "y_test_pred.npy"), allow_pickle=True)
        stats_pred[category] = y_test_pred
    
    auc_bootstraplist = {category: [] for category in ["rhythm", "form", "diagnostic"]}

    for _ in tqdm(range(bootstrap[0])):
        bootstrap_idxes = np.random.choice(mse_losses.shape[0], mse_losses.shape[0], replace=True)
        
        mse_losses_temp, missing_totals_temp = mse_losses[bootstrap_idxes], missing_totals[bootstrap_idxes]
        mse_losses_bootstraplist.append((torch.sum(mse_losses_temp)/torch.sum(missing_totals_temp)).item())

        for category in ["rhythm", "form", "diagnostic"]:
            bootstrap_idxes = np.random.choice(stats_true[category].shape[0], stats_true[category].shape[0], replace=True)
            y_test_true_temp = stats_true[category][bootstrap_idxes]
            y_test_pred_temp = stats_pred[category][bootstrap_idxes]
            try:
                auc_bootstraplist[category].append(roc_auc_score(y_test_true_temp, y_test_pred_temp, average='macro'))
            except ValueError:
                pass
    
    printlog(f"95% CI MSE from Bootstrap {bootstrap}: {2*np.std(mse_losses_bootstraplist)}", path)
    for category in ["rhythm", "form", "diagnostic"]:
        printlog(f"95% CI {category.capitalize()} AUC from Bootstrap {bootstrap}: {2*np.std(auc_bootstraplist[category])}", path)
        np.save(os.path.join(path, f"{category}/auc.npy"), auc_bootstraplist[category])