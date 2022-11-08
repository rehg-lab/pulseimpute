import os
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from datetime import datetime
import re
from tqdm import tqdm

import torch.nn.functional as F
from .brits.mod_utils.brits_dataloader import create_dataloader
from .brits.utils import to_var
from utils.loss_mask import mse_mask_loss

import time
class brits():
    def __init__(self,modelname, rnn_hid_size, impute_weight, label_weight,
                train_data=None, val_data=None, data_name="", 
                 imputation_dict=None, annotate_test="",reload_epoch=-1,recreate=True,
                 annotate="", bs= 64, gpus=[0,1],
                 train_naive=False,
                 train_impute_wind=None, train_impute_prob=None,train_impute_extended=None,
                 train_realppg=None,
                 train_realecg=False, 
                 createjson=True, iter_save=None,
                 brits_data_path=None, bigfile=True, train_time=None):
        
        outpath = "out/"
        if brits_data_path is None:
            brits_data_path = os.path.join(outpath, "..", "models", "brits", "data")
        self.iter_save=iter_save
        self.train_time = train_time
        
        self.bs = bs
        self.gpu_list = gpus
        self.annotate = annotate
        self.annotate_test = annotate_test
        if reload_epoch == -1:
            self.reload_epoch = "latest"
        else:
            self.reload_epoch = reload_epoch
        
        self.dataname=data_name
        
        self.data_loader_setup(train_data, val_data, imputation_dict, recreate=recreate,
                               train_naive=train_naive,
                               train_realppg=train_realppg,
                               train_impute_wind=train_impute_wind, train_impute_prob=train_impute_prob, train_impute_extended=train_impute_extended,
                               train_realecg=train_realecg, createjson=createjson,
                               brits_data_path=brits_data_path, bigfile=bigfile)
        
        # cannot get relative import working for the life of me
        model_module = __import__(f'models.brits.models.{modelname}', fromlist=[""])
        model_module_class = getattr(model_module, "Model")

        if len(self.gpu_list) == 1:
            torch.cuda.set_device(self.gpu_list[0])
        self.model =  nn.DataParallel(model_module_class(rnn_hid_size, 
                                                         impute_weight, 
                                                         label_weight,
                                                         self.total_len,
                                                         self.total_channels),
                                      device_ids=self.gpu_list)

        self.model.to(torch.device(f"cuda:{self.gpu_list[0]}"))

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total params is {}'.format(total_params))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        self.ckpt_path = os.path.join(outpath, data_name+annotate_test, modelname+annotate)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_path, "epoch_latest"), exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_path, "epoch_best"), exist_ok=True)

        self.reload_ckpt_path = os.path.join(outpath, data_name, modelname+annotate)
        self.reload_model()

    def reload_model(self):
        self.epoch_list = [-1]
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.best_val_loss = 9999999
        if os.path.isfile(os.path.join(self.reload_ckpt_path,f"epoch_{self.reload_epoch}", f"epoch_{self.reload_epoch}.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path ,f"epoch_{self.reload_epoch}", f"epoch_{self.reload_epoch}.pkl"), 
                                map_location=f"cuda:{self.gpu_list[0]}")
            self.epoch_list.append(state["epoch"])

            print(f"Reloading given epoch: {np.max(self.epoch_list)}")
            with open(os.path.join(self.reload_ckpt_path, "loss_log.txt"), 'a+') as f:
                f.write(f"Reloading newest epoch: {self.reload_epoch}\n")
            print(self.model.load_state_dict(state['state_dict'], strict=True))
            print(self.optimizer.load_state_dict(state['optimizer']))
            self.bestvalmse = state["val_mse"]
        else:
            print(f"cannot reload epoch {self.reload_epoch}")
            self.bestvalmse = 9999999

    def data_loader_setup(self, train_data, val_data=None, imputation_dict=None, recreate=False, train_naive=False,
    train_impute_wind=None, train_impute_prob=None, train_impute_extended=None,
    train_realppg=None,
    train_realecg=False, createjson=True, brits_data_path = os.path.join("models", "brits", "data"), bigfile=True):
        # num_threads_used = multiprocessing.cpu_count() // 8* len(self.gpu_list)
        num_threads_used = multiprocessing.cpu_count()
        print(f"Num Threads Used: {num_threads_used}")
        torch.set_num_threads(num_threads_used)
        os.environ["MP_NUM_THREADS"]=str(num_threads_used)
        os.environ["OPENBLAS_NUM_THREADS"]=str(num_threads_used)
        os.environ["MKL_NUM_THREADS"]=str(num_threads_used)
        os.environ["VECLIB_MAXIMUM_THREADS"]=str(num_threads_used)
        os.environ["NUMEXPR_NUM_THREADS"]=str(num_threads_used)

        np.random.seed(10)
        # path = os.path.join("models", "brits", "data")

        if val_data is not None:
            if "mimic" in self.dataname:
                prefetch = 1
            else:
                prefetch = 2

            self.train_loader = create_dataloader(data=train_data, dataname=self.dataname, type="train", annotate=self.annotate,
                                            path=brits_data_path, batch_size=self.bs, num_workers=num_threads_used,
                                            train_realppg=train_realppg,
                                            train_impute_wind=train_impute_wind, train_impute_prob=train_impute_prob,train_impute_extended=train_impute_extended,
                                            train_realecg=train_realecg, createjson=createjson, bigfile=bigfile,
                                            prefetch_factor=prefetch)
            self.val_loader = create_dataloader(data=val_data, dataname=self.dataname, type="val", annotate=self.annotate,
                                                path=brits_data_path, batch_size=self.bs, num_workers=num_threads_used,
                                                train_realppg=train_realppg,
                                                train_impute_wind=train_impute_wind, train_impute_prob=train_impute_prob,train_impute_extended=train_impute_extended,
                                                train_realecg=train_realecg, createjson=createjson, bigfile=bigfile,
                                                prefetch_factor=prefetch)
            blah = next(iter(self.train_loader))
            self.total_len = blah["forward"]["values"].shape[1]
            self.total_channels = blah["forward"]["values"].shape[2]
        else:
            self.test_loader = create_dataloader(data=train_data, imputation_dict=imputation_dict, dataname=self.dataname, type="test", annotate=self.annotate,
                                                annotate_test=self.annotate_test,createjson=createjson,
                                                path=brits_data_path, batch_size=self.bs, num_workers=num_threads_used, bigfile=bigfile)
            blah = next(iter(self.test_loader))
            self.total_len = blah["forward"]["values"].shape[1]
            self.total_channels = blah["forward"]["values"].shape[2]
    def testimp(self):
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")

        print(f'{dt_string} | Start')

        with torch.no_grad():
            
            self.model.eval()
            total_test_mse_loss = 0
            total_missing_total=0
            start = 0
            epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(self.reload_epoch))
            os.makedirs(epoch_check_path, exist_ok=True)

            flag = True
            for idx, data in enumerate(tqdm(self.test_loader, desc="Testing", leave=False)):
                data = to_var(data)
                ret = self.model(data)


                eval_masks = ret['eval_masks'].data.cpu().numpy() # this is where the signal is
                eval_ = ret['evals'].data.cpu().numpy()
                target_seq = data["target_seq"]
                target_seq[np.where(eval_masks == 0)] = np.nan

                mse_loss, missing_total = mse_mask_loss(ret["imputations"].to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                        target_seq.to(torch.device(f"cuda:{self.gpu_list[0]}")))
                total_missing_total += missing_total
                total_test_mse_loss += mse_loss.item() 

                if flag:
                    imputation_cat = np.copy(ret['imputations'].data.cpu().numpy())
                    flag=False
                else:
                    imputation_temp = np.copy(ret['imputations'].data.cpu().numpy())
                    imputation_cat = np.concatenate((imputation_cat, imputation_temp), axis=0) 

            self.model.train()
        total_test_mse_loss /= total_missing_total

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | MSE:{total_test_mse_loss:.10f} \n')
        with open(os.path.join(epoch_check_path, "loss_log.txt"), 'a+') as f:
            f.write(f'{dt_string} | MSE:{total_test_mse_loss:.10f} \n')

        np.save(os.path.join(self.ckpt_path, "imputation.npy"), imputation_cat)

        return imputation_cat

    def train(self):
        writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "tb"))
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | Start')
        iter_idx = 0
        if self.iter_save:
            iter_idx = np.max(self.epoch_list)
            start_epoch = 0
        else: 
            if np.max(self.epoch_list) > 0:
                start_epoch = np.max(self.epoch_list)
            else:
                start_epoch = 0

        first_run=True
        start_time = time.time()
        for epoch in range(start_epoch,1000):
            # train set
            self.model.train()
            evals = []
            imputations = []
            if self.iter_save:
                for idx, data  in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
                    data = to_var(data)
                    self.optimizer.zero_grad()
                    ret = self.model(data)
                    ret['loss'].mean().backward()
                    self.optimizer.step()

                    eval_masks = ret['eval_masks'].data.cpu().numpy()
                    eval_ = ret['evals'].data.cpu().numpy()
                    imputation = ret['imputations'].data.cpu().numpy()
                    evals += eval_[np.where(eval_masks == 1)].tolist()
                    imputations += imputation[np.where(eval_masks == 1)].tolist()

                    if not first_run and iter_idx % self.iter_save == 0:
                        evals = np.asarray(evals)
                        imputations = np.asarray(imputations)
                        # import pdb; pdb.set_trace()
                        train_mse = np.square(evals - imputations).mean()
                        writer.add_scalar('mse / Train', train_mse, iter_idx)

                        self.model.eval()

                        if self.dataname == "ptbxl":
                            with torch.no_grad():
                                evals = []
                                imputations = []
                                for idx, data in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):    
                                    data = to_var(data)
                                    ret = self.model(data)

                                    eval_masks = ret['eval_masks'].data.cpu().numpy()
                                    eval_ = ret['evals'].data.cpu().numpy()
                                    imputation = ret['imputations'].data.cpu().numpy()
                                    evals += eval_[np.where(eval_masks == 1)].tolist()
                                    imputations += imputation[np.where(eval_masks == 1)].tolist()

                                    if self.dataname == "mimic" and idx >= 10:
                                        break
                                    
                                evals = np.asarray(evals)
                                imputations = np.asarray(imputations)
                                val_mse = np.square(evals - imputations).mean()
                                writer.add_scalar('mse / Val', val_mse, iter_idx)
                        else:
                            val_mse = 9999
                        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
                        print(f'{dt_string} | Iter: {iter_idx} \nTrain mse: {train_mse:.10f} \nVal mse:{val_mse:.10f} \n')
                        with open(os.path.join(self.ckpt_path, "loss_log.txt"), 'a+') as f:
                            f.write(f'{dt_string} | Iter: {iter_idx} \nTrain mse: {train_mse:.10f} \nVal mse:{val_mse:.10f} \n')

                        evals = []
                        imputations = []

                        if iter_idx != 0:
                            os.remove(os.path.join(self.ckpt_path, "epoch_latest", "epoch_latest.pkl"))
                        state = {   'epoch': iter_idx,
                                    'state_dict': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    "val_mse": val_mse
                                }
                        torch.save(state, os.path.join(self.ckpt_path, "epoch_latest", "epoch_latest.pkl"))

                        if val_mse < self.bestvalmse:
                            if os.path.exists(os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl")):
                                os.remove(os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                            torch.save(state, os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                            self.bestvalmse=val_mse

                        epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(iter_idx))
                        try:
                            os.mkdir(epoch_check_path)
                            torch.save(state, os.path.join(epoch_check_path, "epoch_" +  str(iter_idx) + ".pkl"))
                        except:
                            pass
                    
                    current_time = time.time()
                    # print(f"Time Elapsed in Minutes {(current_time - start_time) / 60}")
                    if current_time - start_time + 60*60*4 >= 60*60*72: # 24 hours
                        with torch.no_grad():
                            evals = []
                            imputations = []
                            for idx, data in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):    
                                data = to_var(data)
                                ret = self.model(data)

                                eval_masks = ret['eval_masks'].data.cpu().numpy()
                                eval_ = ret['evals'].data.cpu().numpy()
                                imputation = ret['imputations'].data.cpu().numpy()
                                evals += eval_[np.where(eval_masks == 1)].tolist()
                                imputations += imputation[np.where(eval_masks == 1)].tolist()

                                if self.dataname == "mimic" and idx >= 10:
                                    break

                            evals = np.asarray(evals)
                            imputations = np.asarray(imputations)
                            val_mse = np.square(evals - imputations).mean()
                            writer.add_scalar('mse / Val', val_mse, iter_idx)
                        epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(iter_idx))
                        try:
                            os.mkdir(epoch_check_path)
                            torch.save(state, os.path.join(epoch_check_path, "epoch_" +  str(iter_idx) + ".pkl"))
                        except:
                            pass
                        break
                    iter_idx += 1
                    first_run = False

            else:
                if self.train_time:
                    if time.time() - start_time >= self.train_time:
                        break
                for idx, data  in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
                    data = to_var(data)
                    self.optimizer.zero_grad()
                    ret = self.model(data)
                    ret['loss'].mean().backward()
                    self.optimizer.step()

                    eval_masks = ret['eval_masks'].data.cpu().numpy()
                    eval_ = ret['evals'].data.cpu().numpy()
                    imputation = ret['imputations'].data.cpu().numpy()
                    evals += eval_[np.where(eval_masks == 1)].tolist()
                    imputations += imputation[np.where(eval_masks == 1)].tolist()


                evals = np.asarray(evals)
                imputations = np.asarray(imputations)
                # import pdb; pdb.set_trace()
                train_mse = np.square(evals - imputations).mean()
                writer.add_scalar('mse / Train', train_mse, epoch)

                # val set
                self.model.eval()
                with torch.no_grad():
                    evals = []
                    imputations = []
                    for idx, data in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):    
                        data = to_var(data)
                        ret = self.model(data)

                        eval_masks = ret['eval_masks'].data.cpu().numpy()
                        eval_ = ret['evals'].data.cpu().numpy()
                        imputation = ret['imputations'].data.cpu().numpy()
                        evals += eval_[np.where(eval_masks == 1)].tolist()
                        imputations += imputation[np.where(eval_masks == 1)].tolist()

                    evals = np.asarray(evals)
                    imputations = np.asarray(imputations)
                    val_mse = np.square(evals - imputations).mean()
                    writer.add_scalar('mse / Val', val_mse, epoch)
                
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
                print(f'{dt_string} | Epoch: {epoch} \nTrain mse: {train_mse:.10f} \nVal mse:{val_mse:.10f} \n')
                with open(os.path.join(self.ckpt_path, "loss_log.txt"), 'a+') as f:
                    f.write(f'{dt_string} | Epoch: {epoch} \nTrain mse: {train_mse:.10f} \nVal mse:{val_mse:.10f} \n')

                if epoch != 0:
                    os.remove(os.path.join(self.ckpt_path, "epoch_latest", "epoch_latest.pkl"))
                state = {   'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            "val_mse": val_mse
                        }
                torch.save(state, os.path.join(self.ckpt_path, "epoch_latest", "epoch_latest.pkl"))

                if val_mse < self.bestvalmse:
                    if os.path.exists(os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl")):
                        os.remove(os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                    torch.save(state, os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                    self.bestvalmse=val_mse

                if epoch % 5 == 0:
                    epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(epoch))
                    if epoch % 5 == 0:
                        try:
                            os.mkdir(epoch_check_path)
                            torch.save(state, os.path.join(epoch_check_path, "epoch_" +  str(epoch) + ".pkl"))
                        except:
                            pass
