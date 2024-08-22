import numpy as np
import os
from tqdm import tqdm
import torch
from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, printlog

def evaluate_mimic(imputation, Y_dict_test, X_test, path, bootstrap, data_type):
    mse_losses, missing_totals = eval_mse(imputation, Y_dict_test["target_seq"], path, return_stats=True)
    printlog(f"MSE: {(torch.sum(mse_losses)/torch.sum(missing_totals)).item()}", path, type="w")
    mse_losses_bootstraplist = []

    stats = eval_heartbeat_detection(imputation=imputation, target_seq=Y_dict_test["target_seq"], 
                                     input=X_test, path=path, return_stats=True)
    sens = np.nanmean(stats[:,0])
    prec = np.nanmean(stats[:,1])
    f1 = 2*sens*prec/(sens+prec)

    printlog(f"F1: {f1}", path)
    printlog(f"Precision: {prec}", path)
    printlog(f"Sensitivity: {sens}", path)
    
    f1_bootstraplist = []
    sens_bootstraplist = []
    prec_bootstraplist = []

    for _ in tqdm(range(bootstrap[0])):
        bootstrap_idxes = np.random.choice(mse_losses.shape[0], mse_losses.shape[0], replace=True)
        
        mse_losses_temp, missing_totals_temp = mse_losses[bootstrap_idxes], missing_totals[bootstrap_idxes]
        mse_losses_bootstraplist.append((torch.sum(mse_losses_temp)/torch.sum(missing_totals_temp)).item())

        stats_temp = stats[bootstrap_idxes]
        sens = np.nanmean(stats_temp[:,0])
        prec = np.nanmean(stats_temp[:,1])
        f1 = 2*sens*prec/(sens+prec)

        f1_bootstraplist.append(f1)
        prec_bootstraplist.append(sens)
        sens_bootstraplist.append(prec)

    printlog(f"95% CI MSE from Bootstrap {bootstrap}: {2*np.std(mse_losses_bootstraplist)}", path)
    printlog(f"95% CI F1 from Bootstrap {bootstrap}: {2*np.std(f1_bootstraplist)}", path)
    printlog(f"95% CI Prec from Bootstrap {bootstrap}: {2*np.std(prec_bootstraplist)}", path)
    printlog(f"95% CI Sens from Bootstrap {bootstrap}: {2*np.std(sens_bootstraplist)}", path)

    np.save(os.path.join(path, "f1.npy"), f1_bootstraplist)
    np.save(os.path.join(path, "precision.npy"), prec_bootstraplist)
    np.save(os.path.join(path, "sensitivity.npy"), sens_bootstraplist)