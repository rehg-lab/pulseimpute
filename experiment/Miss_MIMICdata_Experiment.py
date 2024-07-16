from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, eval_cardiac_classification, printlog
from utils.random_seed import random_seed
from data.PPGMIMICDataset import PPGMIMICDataset
from data.ECGMIMICDataset import ECGMIMICDataset
from data.CustomDataset import CustomDataset
from experiment import PulseImputeExperiment
import numpy as np
import torch
import random
from tqdm import tqdm
import os

class Miss_MIMIC_Experiment:

    def __init__(self, configs, bootstrap):
        self.configs = configs
        self.bootstrap = bootstrap

    
    def train(self):
        config = self.configs # assume we can only train one config in one run
        print(config.modelname+config.annotate)
        random_seed(10, True)
        if 'ppg' in config.data_name:
            dataset_loader = PPGMIMICDataset()
        elif 'ecg' in config.data_name:
            dataset_loader = ECGMIMICDataset()
        else:
            raise Exception("Expected ppg or ecg in data_name.")
        X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader(**{k: v for k, v in config.data_load.__dict__.items() if v is not None}, 
                                                                                train=True, val=True, test=False)
        model_type = config.modeltype
        model_module = __import__(f'models.{model_type}Model_Architecture.{model_type}Model_Wrapper', fromlist=[''])
        model_module_class = getattr(model_module, model_type.lower())
        model = model_module_class(modelname=config.modelname, train_data=X_train, val_data=X_val, 
                                data_name=config.data_name, annotate=config.annotate,  
                                **config.modelparams.__dict__,
                                **config.train.__dict__)
        model.train()

    def test(self):
        for config in self.configs:
            print(config.modelname+config.annotate+config.annotate_test)

            random_seed(10, True)
            if config.data_name == 'custom':
                dataset_loader = CustomDataset(config.data_path)
                # For CustomDataset, we don't pass additional parameters
                X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader.load(train=False, val=False, test=True)
            elif 'ppg' in config.data_name:
                dataset_loader = PPGMIMICDataset()
                X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader.load(**{k: v for k, v in config.data_load.__dict__.items() if v is not None})
            elif 'ecg' in config.data_name:
                dataset_loader = ECGMIMICDataset()
                X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = dataset_loader.load(**{k: v for k, v in config.data_load.__dict__.items() if v is not None})
            else:
                raise Exception("Expected ppg, ecg, or custom in data_name.")
            
            print('LOAD PHASE')


            # If imputation does not exist, call model to create imputation
            print('IMPUTE PHASE')
            path = os.path.join("out/out_test/", config.data_name+config.annotate_test, config.modelname+config.annotate)
            if os.path.exists(os.path.join(path, "imputation.npy")):
                imputation = np.load(os.path.join(path, "imputation.npy"))
            else:
                model_type = config.modeltype
                model_module = __import__(f'models.{model_type}Model_Architecture.{model_type}Model_Wrapper', fromlist=[''])
                model_module_class = getattr(model_module, model_type.lower())
                print(model_type, model_module, model_module_class)
                model = model_module_class(modelname=config.modelname, data_name=config.data_name, 
                                        train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
                                        annotate=config.annotate,  annotate_test=config.annotate_test,  
                                        #modeltype=config['modeltype'],
                                        **config.modelparams.__dict__,
                                        **config.train.__dict__)
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


                for bootstrap_iter in tqdm(range(self.bootstrap[0])):
                    np.random.seed(bootstrap_iter)
                    bootstrap_idxes = np.random.choice(mse_losses.shape[0],mse_losses.shape[0], replace=True)
                    
                    mse_losses_temp, missing_totals_temp = mse_losses[bootstrap_idxes], missing_totals[bootstrap_idxes]
                    mse_losses_bootstraplist.append((torch.sum(mse_losses_temp)/torch.sum(missing_totals_temp)).item())

                    stats_temp = stats[bootstrap_idxes]
                    sens = np.nanmean(stats_temp[:,0])
                    prec = np.nanmean(stats_temp[:,1])
                    f1 = 2*sens*prec/(sens+prec)


                    f1_bootstraplist.append(f1)
                    prec_bootstraplist.append(sens)
                    sens_bootstraplist.append(prec)
                                                    


                
                printlog(f"95% CI MSE from Bootstrap {self.bootstrap}: {2*np.std(mse_losses_bootstraplist)}",path)

                printlog(f"95% CI F1 from Bootstrap {self.bootstrap}: {2*np.std(f1_bootstraplist)}",path)
                printlog(f"95% CI Prec from Bootstrap {self.bootstrap}: {2*np.std(prec_bootstraplist)}",path)
                printlog(f"95% CI Sens from Bootstrap {self.bootstrap}: {2*np.std(sens_bootstraplist)}",path)
                if not os.path.exists(os.path.join(path, "f1.npy")):
                    np.save(os.path.join(path, "f1.npy"), f1_bootstraplist)
                if not os.path.exists(os.path.join(path, "precision.npy")):
                    np.save(os.path.join(path, "precision.npy"), prec_bootstraplist)
                if not os.path.exists(os.path.join(path, "sensitivity.npy")):
                    np.save(os.path.join(path, "sensitivity.npy"), sens_bootstraplist)

            else:
                eval_mse(imputation, Y_dict_test["target_seq"], path)
                if "mimic" in config.data_name:
                    eval_heartbeat_detection(imputation=imputation, target_seq=Y_dict_test["target_seq"], input=X_test, path=path)
                else:
                    eval_cardiac_classification(imputation, path)