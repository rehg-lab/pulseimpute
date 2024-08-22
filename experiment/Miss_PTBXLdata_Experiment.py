from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, eval_cardiac_classification, printlog
from utils.random_seed import random_seed
from data.PTBXLDataset import PTBXLDataset
from experiment import PulseImputeExperiment
import numpy as np
import torch
import random
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score

class Miss_PTBXL_Experiment:

    def __init__(self, config, bootstrap):
        self.config = config
        self.bootstrap = bootstrap

    
    def train(self):
        print(self.config.modelname+self.config.annotate)
        random_seed(10, True)
        dataset_loader = PTBXLDataset()
        X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader(**self.config.data_load, 
                                                                                train=True, val=True, test=False)
        model_type = self.config.modeltype
        model_module = __import__(f'models.{model_type}Model_Architecture.{model_type}Model_Wrapper', fromlist=[''])
        model_module_class = getattr(model_module, model_type.lower())
        model = model_module_class(modelname=self.config.modelname, train_data=X_train, val_data=X_val, 
                                data_type=self.config.data_type, annotate=self.config.annotate,  
                                **self.config.modelparams,
                                **self.config.train)
        model.train()

    def test(self):
        print(self.config.modelname+self.config.annotate+self.config.annotate_test)

        random_seed(10, True)
        dataset_loader = PTBXLDataset()
        #load = getattr(__import__(f'utils.{config["data_type"]}', fromlist=['']), "load")
        # Calls load() from util to load dataset
        print('LOAD PHASE')
        X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader.load(**self.config.data_load)

        # If imputation does not exist, call model to create imputation
        print('IMPUTE PHASE')
        path = os.path.join("out/out_test/", self.config.data_type+self.config.annotate_test, self.config.modelname+self.config.annotate)
        if os.path.exists(os.path.join(path, "imputation.npy")):
            imputation = np.load(os.path.join(path, "imputation.npy"))
        else:
            model_type = self.config.modeltype
            model_module = __import__(f'models.{model_type}Model_Architecture.{model_type}Model_Wrapper', fromlist=[''])
            model_module_class = getattr(model_module, model_type.lower())
            print(model_type, model_module, model_module_class)
            model = model_module_class(modelname=self.config.modelname, data_type=self.config.data_type, 
                                    train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
                                    annotate=self.config.annotate,  annotate_test=self.config.annotate_test,  
                                    #modeltype=config['modeltype'],
                                    **self.config.modelparams,
                                    **self.config.train)
            print(model)
            imputation = model.fit()

        # Also create original and target seq files for the sake of creating visualizations
        if not os.path.exists(os.path.join(path, "original.npy")):
            np.save(os.path.join(path, "original.npy"), X_test)
        if not os.path.exists(os.path.join(path, "target_seq.npy")):
            np.save(os.path.join(path, "target_seq.npy"), Y_dict_test['target_seq'].numpy())

        print('STATS PHASE')
        # getting stats (mostly through utils eval)
        if self.bootstrap is not None: # only support for mimic right now
            mse_losses, missing_totals = eval_mse(imputation, Y_dict_test["target_seq"], path, return_stats=True)
            printlog(f"MSE: {(torch.sum(mse_losses)/torch.sum(missing_totals)).item()}", path, type="w")
            mse_losses_bootstraplist = []

            eval_cardiac_classification(imputation, path)
            stats_true = {}
            stats_pred = {}
            for category in ["rhythm", "form", "diagnostic"]:
                y_test_true = np.load(os.path.join("out/out_test", self.config.data_type+self.config.annotate_test,
                                                            self.config.modelname+self.config.annotate, category,
                                                            "data", "y_test.npy"), allow_pickle=True)
                stats_true[category] = y_test_true
                y_test_pred = np.load(os.path.join("out/out_test", self.config.data_type+self.config.annotate_test,
                                                    self.config.modelname+self.config.annotate, category,
                                                    "models", "fastai_xresnet1d101", "y_test_pred.npy"), 
                                                    allow_pickle=True)
                stats_pred[category] = y_test_pred
            auc_bootstraplist = {
                "rhythm": [],
                "form": [],
                "diagnostic": []  
            }

            for bootstrap_iter in tqdm(range(self.bootstrap[0])):
                np.random.seed(bootstrap_iter)
                bootstrap_idxes = np.random.choice(mse_losses.shape[0],mse_losses.shape[0], replace=True)
                
                mse_losses_temp, missing_totals_temp = mse_losses[bootstrap_idxes], missing_totals[bootstrap_idxes]
                mse_losses_bootstraplist.append((torch.sum(mse_losses_temp)/torch.sum(missing_totals_temp)).item())

                for category in ["rhythm", "form", "diagnostic"]:
                    np.random.seed(bootstrap_iter)
                    bootstrap_idxes = np.random.choice(stats_true[category].shape[0],  
                                                        stats_true[category].shape[0], replace=True)
                    y_test_true_temp = stats_true[category][bootstrap_idxes]
                    y_test_pred_temp = stats_pred[category][bootstrap_idxes]
                    try:
                        auc_bootstraplist[category].append(roc_auc_score(y_test_true_temp, 
                                                                        y_test_pred_temp, 
                                                                        average='macro'))
                    except ValueError:
                        pass
                    
            printlog(f"95% CI MSE from Bootstrap {self.bootstrap}: {2*np.std(mse_losses_bootstraplist)}",path)

            printlog(f"95% CI Rhy AUC from Bootstrap {self.bootstrap}: {2*np.std(auc_bootstraplist['rhythm'])}",path)
            printlog(f"95% CI Form AUC from Bootstrap {self.bootstrap}: {2*np.std(auc_bootstraplist['form'])}",path)
            printlog(f"95% CI Diag AUC from Bootstrap {self.bootstrap}: {2*np.std(auc_bootstraplist['diagnostic'])}",path)
            if not os.path.exists(os.path.join(path, "diagnostic/auc.npy")):
                np.save(os.path.join(path, "diagnostic/auc.npy"), auc_bootstraplist['diagnostic'])
            if not os.path.exists(os.path.join(path, "rhythm/auc.npy")):
                np.save(os.path.join(path, "rhythm/auc.npy"), auc_bootstraplist['rhythm'])
            if not os.path.exists(os.path.join(path, "form/auc.npy")):
                np.save(os.path.join(path, "form/auc.npy"), auc_bootstraplist['form'])

        else:
            eval_mse(imputation, Y_dict_test["target_seq"], path)
            if "mimic" in self.config.data_type:
                eval_heartbeat_detection(imputation=imputation, target_seq=Y_dict_test["target_seq"], input=X_test, path=path)
            else:
                eval_cardiac_classification(imputation, path)