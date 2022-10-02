import os
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm

import time

from utils.loss_mask import mse_mask_loss
from .transformer.utils.misc_utils import l2_mpc_loss, transformer_to_longformer, reload_model, mpc_dataset


class transformer():
    def __init__(self, modelname, data_name, train_data=None, val_data=None, 
                imputation_dict= None, annotate="",annotate_test="",
                 bs= 64, gpus=[0,1], train_time=999, # in hours
                 train_impute_wind=None, train_impute_prob=None, train_impute_extended=None,
                 train_realecg=False, train_realppg=False,
                 max_len=None, iter_save=None,
                 reload_epoch=-1,
                 convertolong=None, reload_epoch_long=None):

        outpath = "out/"

        self.iter_save = iter_save
        self.train_time = train_time
        self.data_name = data_name
        self.bs = bs
        self.gpu_list = gpus
        self.train_realppg = train_realppg
        self.train_realecg = train_realecg
        self.train_impute_extended = train_impute_extended
        self.train_impute_wind = train_impute_wind
        self.train_impute_prob = train_impute_prob

        if reload_epoch == -1:
            self.reload_epoch = "latest"
        else:
            self.reload_epoch = reload_epoch

        self.data_loader_setup(train_data, val_data, imputation_dict=imputation_dict)

        model_module = __import__(f'models.transformer.{modelname}', fromlist=[""])
        model_module_class = getattr(model_module, "MainModel")
        if len(self.gpu_list) == 1:
            torch.cuda.set_device(self.gpu_list[0])

        self.model =  nn.DataParallel(model_module_class(orig_dim=self.total_channels, max_len=max_len), device_ids=self.gpu_list)

        self.model.to(torch.device(f"cuda:{self.gpu_list[0]}"))
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total params is {}'.format(total_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)

        self.ckpt_path = os.path.join(outpath, data_name+annotate_test, modelname+annotate)
        self.reload_ckpt_path = os.path.join(outpath, data_name, modelname+annotate)
        self.epoch_list, self.best_val_loss = reload_model(self.model, self.optimizer, self.reload_epoch, self.reload_ckpt_path, gpu=self.gpu_list[0])

        if convertolong:
            transformer_to_longformer(self.model, modelname = modelname, converttolong_dict=convertolong, gpu=self.gpu_list[0])

            if reload_epoch_long: # this is where we are loading a longformer epoch
                state = torch.load(os.path.join(self.reload_ckpt_path ,f"epoch_{reload_epoch_long}", f"epoch_{reload_epoch_long}.pkl"), 
                                map_location=f"cuda:{self.gpu_list[0]}")
                self.epoch_list.append(state["epoch"])

                print(f"Reloading given epoch: {np.max(self.epoch_list)}")
                with open(os.path.join(self.reload_ckpt_path, "loss_log.txt"), 'a+') as f:
                    f.write(f"Reloading newest epoch: {self.reload_epoch}\n")
                print(self.model.load_state_dict(state['state_dict'], strict=True))



    def data_loader_setup(self, train_data, val_data=None, imputation_dict=None):
        num_threads_used = multiprocessing.cpu_count() # can divide this by number of gpus to allocate threads 
        print(f"Num Threads Used: {num_threads_used}")
        torch.set_num_threads(num_threads_used)
        os.environ["MP_NUM_THREADS"]=str(num_threads_used)
        os.environ["OPENBLAS_NUM_THREADS"]=str(num_threads_used)
        os.environ["MKL_NUM_THREADS"]=str(num_threads_used)
        os.environ["VECLIB_MAXIMUM_THREADS"]=str(num_threads_used)
        os.environ["NUMEXPR_NUM_THREADS"]=str(num_threads_used)

        if val_data is not None:
            train_dataset = mpc_dataset(waveforms=train_data, 
                                        train_impute_wind=self.train_impute_wind, train_impute_prob=self.train_impute_prob, 
                                        train_impute_extended=self.train_impute_extended,
                                        train_realppg=self.train_realppg,
                                        train_realecg=self.train_realecg, type="train")
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=num_threads_used)
            temp = next(iter(self.train_loader))
            val_dataset = mpc_dataset(waveforms=val_data, 
                                        train_impute_wind=self.train_impute_wind, train_impute_prob=self.train_impute_prob, 
                                        train_impute_extended=self.train_impute_extended,
                                        train_realppg=self.train_realppg,
                                        train_realecg=self.train_realecg, type="val")
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
        else:
            test_dataset = mpc_dataset(waveforms=train_data, imputation_dict=imputation_dict,
                                        train_impute_wind=None, train_impute_prob=None, 
                                        train_impute_extended=None,
                                        train_realppg=False,
                                        train_realecg=False, type="test")
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
            temp = next(iter(self.test_loader))

        self.total_channels = temp[0].shape[2]


    def testimp(self):
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")

        print(f'{dt_string} | Start')

        with torch.no_grad():
            self.model.eval()
            total_test_mse_loss = 0
            total_missing_total = 0

            flag=True
            for local_batch, local_label_dict in tqdm(self.test_loader, desc="Testing", leave=False):  
                # 1s at normal vals 0s at mask vals
                mpc_projection = self.model(local_batch)
                mse_loss,missing_total = mse_mask_loss(mpc_projection.to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                            local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))
                total_test_mse_loss += mse_loss.item()
                total_missing_total += missing_total

                mpc_projection[torch.where(torch.isnan(local_label_dict["target_seq"]))] = \
                    local_batch[torch.where(torch.isnan(local_label_dict["target_seq"]))].cuda()

                if flag:
                    flag = False
                    imputation_cat = np.copy(mpc_projection.cpu().detach().numpy())
                else:
                    imputation_temp = np.copy(mpc_projection.cpu().detach().numpy())
                    imputation_cat = np.concatenate((imputation_cat, imputation_temp), axis=0)
                # break


            self.model.train()
        total_test_mse_loss /= total_missing_total

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | MSE:{total_test_mse_loss:.10f}  \n')
        with open(os.path.join(self.ckpt_path, "loss_log.txt"), 'a+') as f:
            f.write(f'{dt_string} | MSE:{total_test_mse_loss:.10f}  \n')

        np.save(os.path.join(self.ckpt_path, "imputation.npy"), imputation_cat)

        return imputation_cat


    def train(self):

        writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "tb"))

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | Start')

        start = time.time()

        iter_idx = 0
        for epoch in range(np.max(self.epoch_list)+1, 1000):
            total_train_mpcl2_loss = 0
            total_missing_total = 0

            if self.iter_save: # for saving based on iterations, needed for MIMIC datasets
                
                for local_batch, local_label_dict in tqdm(self.train_loader, desc="Training", leave=False):
                    iter_idx += 1
                    self.optimizer.zero_grad()
                    mpc_projection = self.model(local_batch)
                    mpcl2_loss = l2_mpc_loss(mpc_projection.to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                                local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))

                    mpcl2_loss.backward()

                    self.optimizer.step()
                    total_train_mpcl2_loss += mpcl2_loss.item()
                    total_missing_total += torch.sum(~torch.isnan(local_label_dict["target_seq"]))
                    

                    if iter_idx % self.iter_save == 0:

                        total_train_mpcl2_loss /= total_missing_total
                        writer.add_scalar('Masked Predictive Coding L2 Loss/Train', total_train_mpcl2_loss, iter_idx)

                        with torch.no_grad():
                            self.model.eval()
                            total_val_mpcl2_loss = 9999
                            total_missing_total = 0

                            val_iter_idx = 0
                            for local_batch, local_label_dict in tqdm(self.val_loader, desc="Validating", leave=False): 
                                val_iter_idx += 1

                                mpc_projection = self.model(local_batch)

                                mpcl2_loss = l2_mpc_loss(mpc_projection.to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                                        local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))
                                total_val_mpcl2_loss += mpcl2_loss.item()
                                total_missing_total += torch.sum(~torch.isnan(local_label_dict["target_seq"]))
                                if val_iter_idx >= self.iter_save/10:
                                    break

                                
                            self.model.train()
                        total_val_mpcl2_loss /= total_missing_total
                        writer.add_scalar('Masked Predictive Coding L2 Loss/Val', total_val_mpcl2_loss, iter_idx)

                        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
                        print(f'{dt_string} | iter_idx: {iter_idx} \nTrain MPC L2 Loss: {total_train_mpcl2_loss:.8f} \nVal MPC L2 Loss:{total_val_mpcl2_loss:.8f} \n')
                        with open(os.path.join(self.ckpt_path, "loss_log.txt"), 'a+') as f:
                            f.write(f'{dt_string} | iter_idx: {iter_idx} \nTrain MPC L2 Loss: {total_train_mpcl2_loss:.8f} \nVal MPC L2 Loss:{total_val_mpcl2_loss:.8f} \n')

                        state = {   'epoch': iter_idx,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                "l2valloss": total_val_mpcl2_loss,
                        }
                        epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(iter_idx))
                        try:
                            os.mkdir(epoch_check_path)
                            torch.save(state, os.path.join(epoch_check_path, "epoch_" +  str(iter_idx) + ".pkl"))
                        except:
                            pass

                        if total_val_mpcl2_loss <= self.best_val_loss:
                            try:
                                os.remove(os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                            except OSError:
                                pass
                            torch.save(state, os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                            self.best_val_loss = total_val_mpcl2_loss

                        total_train_mpcl2_loss = 0
                        total_missing_total = 0

            else: # for saving based on epochs 
                for local_batch, local_label_dict in tqdm(self.train_loader, desc="Training", leave=False):
                    end = time.time()
                    if (end-start) / 60 / 60 >= self.train_time:
                        import sys; sys.exit()

                    self.optimizer.zero_grad()
                    mpc_projection = self.model(local_batch)
                    mpcl2_loss = l2_mpc_loss(mpc_projection.to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                                   local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))

                    mpcl2_loss.backward()


                    self.optimizer.step()
                    total_train_mpcl2_loss += mpcl2_loss.item()
                    total_missing_total += torch.sum(~torch.isnan(local_label_dict["target_seq"]))


                total_train_mpcl2_loss /= total_missing_total
                writer.add_scalar('Masked Predictive Coding L2 Loss/Train', total_train_mpcl2_loss, epoch)

                with torch.no_grad():
                    self.model.eval()
                    total_val_mpcl2_loss = 0
                    total_missing_total = 0

                    residuals_all = []
                    for local_batch, local_label_dict in tqdm(self.val_loader, desc="Validating", leave=False): 
                        mpc_projection = self.model(local_batch)
                        mpcl2_loss, residuals = l2_mpc_loss(mpc_projection.to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                                local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")), residuals=True)
                        residuals_all.append(residuals)
                        total_val_mpcl2_loss += mpcl2_loss.item()
                        total_missing_total += torch.sum(~torch.isnan(local_label_dict["target_seq"]))

                    self.model.train()
                total_val_mpcl2_loss /= total_missing_total

                total_val_mpcl2_std = torch.std(torch.cat(residuals_all))
                writer.add_scalar('Masked Predictive Coding L2 Loss/Val', total_val_mpcl2_loss, epoch)

                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
                print(f'{dt_string} | Epoch: {epoch}\nTrain MPC L2 Loss: {total_train_mpcl2_loss:.8f} \nVal MPC L2 Loss:{total_val_mpcl2_loss:.8f} Val MPC L2 std: {total_val_mpcl2_std:.8f}\n')
                with open(os.path.join(self.ckpt_path, "loss_log.txt"), 'a+') as f:
                    f.write(f'{dt_string} | Epoch: {epoch} \nTrain MPC L2 Loss: {total_train_mpcl2_loss:.8f} \nVal MPC L2 Loss:{total_val_mpcl2_loss:.8f} Val MPC L2 std: {total_val_mpcl2_std:.8f} \n')

                if epoch != 0:
                    os.remove(os.path.join(self.ckpt_path, "epoch_latest", "epoch_latest.pkl"))
                state = {   'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            "l2valloss": total_val_mpcl2_loss,
                            }
                torch.save(state, os.path.join(self.ckpt_path, "epoch_latest", "epoch_latest.pkl"))
                if total_val_mpcl2_loss <= self.best_val_loss:
                    if epoch != 0:
                        os.remove(os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                    torch.save(state, os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                    self.best_val_loss = total_val_mpcl2_loss

                if "ptbxl" in self.data_name:
                    save_epoch = 50
                else:
                    save_epoch = 1
                epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(epoch))
                if epoch % save_epoch == 0:
                    try:
                        os.mkdir(epoch_check_path)
                        torch.save(state, os.path.join(epoch_check_path, "epoch_" +  str(epoch) + ".pkl"))
                    except:
                        pass

        writer.close()


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params

