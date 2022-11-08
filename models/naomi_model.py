import os
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from datetime import datetime
import time
from .NAOMI.helpers import update_discrim, update_policy
from .NAOMI.model import Discriminator
from .NAOMI.naomi_policy import naomi_policy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from tqdm import tqdm


from utils.loss_mask import mse_mask_loss


import gc
import csv
from ast import literal_eval



class naomi():
    def __init__(self,modelname, data_name, train_data=None, val_data=None, 
                 imputation_dict= None, annotate="",annotate_test="",gpus=[0,1], 
                 params=None, pretrain_epochs=10, pretrain_iters=None, val_iters=None,
                 clip =10,
                 policy_learning_rate=1e-6, discrim_learning_rate=1e-3, pre_start_lr=1e-3,
                 pretrain_disc_iter = 2000, max_iter_num=60000,
                 save_model_interval=500, log_interval=100,
                 train_impute_wind=None, train_impute_prob=None, train_impute_extended=None,
                 train_realecg=False, train_realppg=False,
                 reload_epoch=-1, save_batches = None):
        outpath = "out/"
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_iters = pretrain_iters
        self.val_iters=val_iters
        self.clip=clip
        self.policy_learning_rate = policy_learning_rate
        self.discrim_learning_rate=discrim_learning_rate
        self.pretrain_disc_iter = pretrain_disc_iter
        self.max_iter_num = max_iter_num
        self.log_interval = log_interval
        self.save_model_interval = save_model_interval
        self.pre_start_lr = pre_start_lr
        self.save_batches = save_batches


        self.data_name = data_name
        self.bs = params["batch"]
        params["batch"] = int(params["batch"] / len(gpus))
        self.gpu_list = gpus
        self.train_realppg = train_realppg
        self.train_realppg= train_realppg
        if train_realecg or train_realppg:
            if train_realppg:
                tuples_path_train = os.path.join("data", "missingness_patterns", f"missing_ppg_train.csv")
                tuples_path_val = os.path.join("data", "missingness_patterns", f"missing_ppg_val.csv")
            elif train_realecg:
                tuples_path_train = os.path.join("data", "missingness_patterns", f"missing_ecg_train.csv")
                tuples_path_val = os.path.join("data", "missingness_patterns", f"missing_ecg_val.csv")

            with open(tuples_path_train, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                self.list_of_miss_train = list(csv_reader)
            with open(tuples_path_val, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                self.list_of_miss_val = list(csv_reader)
        else:
            self.list_of_miss_train = None
            self.list_of_miss_val = None
        self.train_impute_extended=train_impute_extended
        self.train_impute_wind = train_impute_wind
        self.train_impute_prob = train_impute_prob
        self.annotate_test = annotate_test
        if reload_epoch == -1:
            self.reload_epoch = "latest"
        else:
            self.reload_epoch = reload_epoch

        self.data_loader_setup(train_data, val_data, imputation_dict=imputation_dict)

        if len(self.gpu_list) == 1:
            torch.cuda.set_device(self.gpu_list[0])
        self.naomi_policy =  nn.DataParallel(naomi_policy(params), 
                                    device_ids=self.gpu_list)
        self.naomi_policy.to(torch.device(f"cuda:{self.gpu_list[0]}"))
 
        self.naomi_discrim =  nn.DataParallel(Discriminator(params), 
                                    device_ids=self.gpu_list)
        self.naomi_discrim.to(torch.device(f"cuda:{self.gpu_list[0]}"))

        self.optimizer_policy = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.naomi_policy.parameters()),
            lr=self.policy_learning_rate)
        self.optimizer_discrim = torch.optim.Adam(self.naomi_discrim.parameters(), 
            lr=self.discrim_learning_rate)
        self.discrim_criterion = nn.BCELoss()

        total_params = sum(p.numel() for p in self.naomi_policy.parameters() if p.requires_grad) + sum(p.numel() for p in self.naomi_discrim.parameters() if p.requires_grad)
        print('Total params is {}'.format(total_params))

        self.ckpt_path = os.path.join(outpath, data_name+annotate_test, modelname+annotate)
        self.reload_ckpt_path = os.path.join(outpath, data_name, modelname+annotate)
        self.reload_model()
    def printlog(self, line):
        print(line)
        with open(os.path.join(self.ckpt_path, 'loss_log.txt'), 'a') as file:
            file.write(line+'\n')
    def pretrain_policy(self, writer=None):
        self.best_val_loss = 99999
        lr = self.pre_start_lr
        if self.pretrain_iters:
            pretrain_iter = 0
            with tqdm(total= self.pretrain_iters, leave=False) as pbar:
                while pretrain_iter < self.pretrain_iters:
                    total_train_loss = 0
                    self.naomi_policy.train()
                    for local_batch, local_label_dict in tqdm(self.train_loader, "train", leave=False):
                        if pretrain_iter == self.pretrain_iters // 2:
                            lr = lr / 10

                        local_batch, mask = create_missingness(local_batch, self.train_impute_wind, self.train_impute_prob, self.train_impute_extended, self.list_of_miss_train)
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, self.naomi_policy.parameters()),
                            lr=lr)
                        batch_loss = self.naomi_policy(local_batch, local_label_dict["original"], forward=True)
                        total_train_loss += batch_loss.item()

                        optimizer.zero_grad()
                        batch_loss.backward()
                        nn.utils.clip_grad_norm_(self.naomi_policy.parameters(), self.clip)
                        optimizer.step()

                        running_train_loss = total_train_loss / (self.train_loader.batch_size * (pretrain_iter + 1))

                        self.printlog(f'Epoch {pretrain_iter}: Train Loss {running_train_loss}')
                        if writer:
                            writer.add_scalar('MSE Loss/PretrainTeacher', running_train_loss, pretrain_iter)
                        
                        if pretrain_iter >= self.pretrain_iters:
                            break
                        pretrain_iter += 1

            filename = os.path.join(self.ckpt_path, "policy_step_state_dict_best_pretrain.pth")
            torch.save(self.naomi_policy.state_dict(), filename)
            self.printlog('Best model at epoch '+str(pretrain_iter))
        else:
            for e in tqdm(range(self.pretrain_epochs), "pretraining", leave=False):
                epoch = e+1
                if epoch == self.pretrain_epochs // 2:
                    lr = lr / 10

                total_train_loss = 0
                self.naomi_policy.train()
                for local_batch, local_label_dict in tqdm(self.train_loader, "train", leave=False):
                    local_batch, mask = create_missingness(local_batch, self.train_impute_wind, self.train_impute_prob, self.train_impute_extended, self.list_of_miss_train)
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, self.naomi_policy.parameters()),
                        lr=lr)
                    batch_loss = self.naomi_policy(local_batch, local_label_dict["original"], forward=True)
                    total_train_loss += batch_loss.item()

                    optimizer.zero_grad()
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(self.naomi_policy.parameters(), self.clip)
                    optimizer.step()
                total_train_loss /= len(self.train_loader)
                self.printlog(f'Epoch {e}: Train Loss {total_train_loss}')
                if writer:
                    writer.add_scalar('MSE Loss/PretrainTeacher', total_train_loss, e)

                total_val_loss = 0
                self.naomi_policy.eval()
                with torch.no_grad():
                    for local_batch, local_label_dict in tqdm(self.val_loader, "val", leave=False):
                        local_batch, mask = create_missingness(local_batch, self.train_impute_wind, self.train_impute_prob, self.train_impute_extended, self.list_of_miss_val)
                        batch_loss = self.naomi_policy(local_batch, local_label_dict["original"], forward=True)
                        total_val_loss += batch_loss.item()
                    total_val_loss /= len(self.val_loader)
                self.printlog(f'Epoch {e}: Val Loss {total_val_loss}')

                if total_val_loss < self.best_val_loss:    
                    self.best_val_loss = total_val_loss
                    filename = os.path.join(self.ckpt_path, "policy_step_state_dict_best_pretrain.pth")
                    torch.save(self.naomi_policy.state_dict(), filename)
                    self.printlog('Best model at epoch '+str(epoch))

                filename = os.path.join(self.ckpt_path, "policy_step_state_dict_latest_pretrain.pth")
                torch.save(self.naomi_policy.state_dict(), filename)
                self.printlog('Saved model')

        print(f"Finished Pretraining, Best Val Loss {self.best_val_loss}")
    
    def pretrain_discrim(self, writer=None):
        disc_iter = 0
        with tqdm(total= self.pretrain_disc_iter, leave=False) as pbar:
            while disc_iter < self.pretrain_disc_iter:
                for local_batch, local_label_dict in self.train_loader:
                    local_batch, mask = create_missingness(local_batch, self.train_impute_wind, self.train_impute_prob, self.train_impute_extended, self.list_of_miss_train)
                    _, _, _, _, _, model_seq, _, _ = \
                        self.naomi_policy(local_batch, local_label_dict["original"], sample=True)
                    # exp_states is the first 999 steps and exp actions is the last 999 steps of the original seq 
                    with torch.no_grad():
                        mask = np.transpose(mask, (1,0,2)) == 0 # mask now for missingness 
                        mseloss = nn.MSELoss()(model_seq[mask], local_label_dict["original"].transpose(0,1)[mask].cuda())
                    pre_mod_p, pre_exp_p, d_loss = update_discrim(self.naomi_discrim, self.optimizer_discrim, self.discrim_criterion, 
                        local_label_dict["original"].transpose(0,1)[:-1, :, :], 
                        local_label_dict["original"].transpose(0,1)[1:, :, :],
                        model_seq[:-1, :, :], 
                        model_seq[1:, :, :], None, dis_times=3.0, use_gpu=True, train=True)
                    
                    if disc_iter % self.log_interval == 0:
                        self.printlog(f"\n Discrim Pretrain Iter: {disc_iter}, D: {d_loss}, MSE: {mseloss.item()}, exp: , {pre_exp_p.item()},mod: {pre_mod_p.item()}")
                        if writer:
                            writer.add_scalar('MSE Loss/PretrainTeacher', mseloss.item(), disc_iter)
                            writer.add_scalar('MSE Loss/PretrainDiscrim', d_loss, disc_iter)

                    if pre_mod_p < 0.3 or disc_iter >= self.pretrain_disc_iter:
                        disc_iter = self.pretrain_disc_iter
                        break
                    disc_iter += 1
                    pbar.update(1)

                    del model_seq
                    del pre_mod_p
                    del pre_exp_p
                    del mask 
                    del local_batch
                    del local_label_dict
                    del mseloss
                    gc.collect()
        torch.save(self.naomi_policy.state_dict(), os.path.join(self.ckpt_path, "policy_step_pretrained.pth"))
        torch.save(self.naomi_discrim.state_dict(), os.path.join(self.ckpt_path, "discrim_step_pretrained.pth"))

    def testimp(self):
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")

        print(f'{dt_string} | Start')

        with torch.no_grad():
            self.naomi_policy.eval()
            total_test_mse_loss = 0
            total_missing_total = 0

            flag=True
            idx = 0
            start = 0
            epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(self.reload_epoch))
            os.makedirs(epoch_check_path, exist_ok=True)

            for local_batch, local_label_dict in tqdm(self.test_loader, desc="Testing", leave=True):  
                _, _, _, _, _, model_seq, _, _ = \
                                    self.naomi_policy(local_batch,local_label_dict["original"], sample=True) # local_label_dict["original"], sample=True)
                mpc_projection = model_seq.transpose(0,1)
                mse_loss,missing_total = mse_mask_loss(mpc_projection.to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                            local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))
                total_test_mse_loss += mse_loss.item()
                total_missing_total += missing_total

                if flag:
                    flag = False
                    imputation_cat = np.copy(mpc_projection.cpu().detach().numpy())
                else:
                    imputation_temp = np.copy(mpc_projection.cpu().detach().numpy())
                    imputation_cat = np.concatenate((imputation_cat, imputation_temp), axis=0)

                del model_seq
                # del mpc_projection 
                del mse_loss    
                gc.collect()
                if self.save_batches:
                    if idx - start >= 10: # first not inclusive, unless it is the first one
                        flag=True
                        np.save(os.path.join(epoch_check_path, f"{self.save_batches}.imputation_{start*self.bs}_{idx*self.bs}.npy"), imputation_cat)
                        start = idx 
 
                idx += 1
                # if idx >= 4:
                #     break
            self.naomi_policy.train()
        total_test_mse_loss /= total_missing_total

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | MSE:{total_test_mse_loss:.10f} \n')
        with open(os.path.join(epoch_check_path, "loss_log.txt"), 'a+') as f:
            f.write(f'{dt_string} | MSE:{total_test_mse_loss:.10f} \n')
        if self.save_batches:
            np.save(os.path.join(epoch_check_path, f"{self.save_batches}.imputation_{start*self.bs}_{idx*self.bs}.npy"), imputation_cat)
        else:
            np.save(os.path.join(self.ckpt_path, "imputation.npy"), imputation_cat)

        return imputation_cat

        

    def train(self):

        writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "tb"))

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | Start')
        if self.pretrain_epochs > 0 or self.pretrain_iters > 0:
            self.pretrain_policy(writer)
        if self.pretrain_disc_iter > 0 :
            print(self.naomi_policy.load_state_dict(torch.load(os.path.join(self.ckpt_path, "policy_step_state_dict_best_pretrain.pth"))))
            self.pretrain_discrim(writer)
        
        # try:
        #     print(self.naomi_policy.load_state_dict(torch.load(os.path.join(self.ckpt_path, "policy_step_pretrained.pth"))))
        # except FileNotFoundError:
        #     print(os.path.join(self.ckpt_path, "policy_step_pretrained.pth") + " is not found, so we cannot reload. Continuing anyways")
        # try:
        #     print(self.naomi_discrim.load_state_dict(torch.load(os.path.join(self.ckpt_path, "discrim_step_pretrained.pth"))))
        # except FileNotFoundError:
        #     print(os.path.join(self.ckpt_path, "discrim_step_pretrained.pth") + " is not found, so we cannot reload. Continuing anyways")

        iter_idx = 0
        if np.max(self.epoch_list) > 0:
            iter_idx = np.max(self.epoch_list) 
        best_val_mseloss = 99999
        with tqdm(total= self.max_iter_num, leave=False) as pbar:
            while iter_idx < self.max_iter_num:
                for local_batch, local_label_dict in tqdm(self.train_loader, desc="Training", leave=False):
                    local_batch, mask = create_missingness(local_batch, self.train_impute_wind, self.train_impute_prob, self.train_impute_extended, self.list_of_miss_train)
                    ts0 = time.time()
                    _, _, _, _, _, model_seq, _, _ = \
                        self.naomi_policy(local_batch, local_label_dict["original"], sample=True)
                    # exp_states is the first 999 steps and exp actions is the last 999 steps of the original seq 
                    with torch.no_grad():
                        mask = np.transpose(mask, (1,0,2)) == 0 # mask now for missingness 
                        mseloss = nn.MSELoss()(model_seq[mask], local_label_dict["original"].transpose(0,1)[mask].cuda())

                    ts1 = time.time()

                    t0 = time.time()
                    mod_p_epoch, exp_p_epoch, d_loss = update_discrim(self.naomi_discrim, self.optimizer_discrim, self.discrim_criterion, 
                        local_label_dict["original"].transpose(0,1)[:-1, :, :], 
                        local_label_dict["original"].transpose(0,1)[1:, :, :],
                        model_seq[:-1, :, :], 
                        model_seq[1:, :, :], None, dis_times=3.0, use_gpu=True, train=True)
                    
                    writer.add_scalar('MSE Loss/Train', mseloss.item(), iter_idx)
                    writer.add_scalar('Mod P/Train', mod_p_epoch.item(), iter_idx)
                    writer.add_scalar('Exp P/Train', exp_p_epoch.item(), iter_idx)
                    writer.add_scalar('D Loss/Train', d_loss, iter_idx)


                    if iter_idx > 3 and mod_p_epoch < 0.8:
                        update_policy(self.naomi_policy, self.optimizer_policy, self.naomi_discrim, 
                        self.discrim_criterion, model_seq[:-1, :, :], model_seq[1:, :, :], None, use_gpu=True) 
                    t1 = time.time()

                    if iter_idx % self.log_interval == 0:
                        self.printlog('\n Train Iter num{}\tT_sample {:.4f}\tT_update {:.4f}\tD Loss: {:.8f} \tMSE: {:.8f}\texp_p {:.3f}\tmod_p {:.3f}'.format(
                            iter_idx, ts1-ts0, t1-t0, d_loss, mseloss.item(),exp_p_epoch.item(), mod_p_epoch.item()))
                        
                        self.naomi_policy.eval()
                        with torch.no_grad():
                            total_val_mseloss = 0
                            val_iter = 0
                            for local_batch, local_label_dict in tqdm(self.val_loader, desc="Validating", leave=False):
                                if self.val_iters is not None:
                                    if val_iter >= self.val_iters:
                                        break
                                local_batch, mask = create_missingness(local_batch, self.train_impute_wind, self.train_impute_prob, self.train_impute_extended, self.list_of_miss_val)
                                _, _, _, _, _, model_seq, _, _ = \
                                    self.naomi_policy(local_batch, local_label_dict["original"], sample=True)
                                # exp_states is the first 999 steps and exp actions is the last 999 steps of the original seq 
                                mask = np.transpose(mask, (1,0,2)) == 0 # mask now for missingness 
                                mseloss = nn.MSELoss()(model_seq[mask], local_label_dict["original"].transpose(0,1)[mask].cuda())
                                total_val_mseloss += mseloss.item()
                                val_iter += 1
                            if self.val_iters is not None:
                                total_val_mseloss /= self.val_iters
                            else:
                                total_val_mseloss /= len(self.val_loader)
                            self.printlog(f'Val Iter num:{iter_idx} \t  MSE: {total_val_mseloss}')
                            writer.add_scalar('MSE Loss/Val', total_val_mseloss, iter_idx)
                            if total_val_mseloss <= best_val_mseloss:
                                state = {   'iter': iter_idx,
                                    'policy_state_dict': self.naomi_policy.state_dict(),
                                    'discrim_state_dict': self.naomi_discrim.state_dict(),
                                    'optimizer_policy': self.optimizer_policy.state_dict(),
                                    'optimizer_discrim': self.optimizer_discrim.state_dict(),
                                    "msevalloss": total_val_mseloss
                                    }
                                torch.save(state, os.path.join(self.ckpt_path, "epoch_best", f"epoch_best.pkl"))
                                best_val_mseloss = total_val_mseloss
                        self.naomi_policy.train()

                        if (iter_idx) % self.save_model_interval == 0:
                            state = {   'iter': iter_idx,
                                        'policy_state_dict': self.naomi_policy.state_dict(),
                                        'discrim_state_dict': self.naomi_discrim.state_dict(),
                                        'optimizer_policy': self.optimizer_policy.state_dict(),
                                        'optimizer_discrim': self.optimizer_discrim.state_dict(),
                                        "msevalloss": total_val_mseloss
                                        }
                            epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(iter_idx))
                            os.mkdir(epoch_check_path)
                            torch.save(state, os.path.join(epoch_check_path, f"epoch_{iter_idx}.pkl"))

                    iter_idx += 1
                    pbar.update(1)
                    del model_seq
                    del mod_p_epoch
                    del exp_p_epoch
                    del local_batch
                    del local_label_dict
                    del mseloss
                    gc.collect()
        writer.close()

    def data_loader_setup(self, train_data, val_data=None, imputation_dict=None):
        num_threads_used = multiprocessing.cpu_count() 
        # num_threads_used = multiprocessing.cpu_count() // 4 * 2
        print(f"Num Threads Used: {num_threads_used}")
        torch.set_num_threads(num_threads_used)
        os.environ["MP_NUM_THREADS"]=str(num_threads_used)
        os.environ["OPENBLAS_NUM_THREADS"]=str(num_threads_used)
        os.environ["MKL_NUM_THREADS"]=str(num_threads_used)
        os.environ["VECLIB_MAXIMUM_THREADS"]=str(num_threads_used)
        os.environ["NUMEXPR_NUM_THREADS"]=str(num_threads_used)

        if val_data is not None:
            train_dataset = pulseimpute_dataset(waveforms=train_data)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=num_threads_used, drop_last=True)

            val_dataset = pulseimpute_dataset(waveforms=val_data)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used, drop_last=True)
        else:
            test_dataset = pulseimpute_dataset(waveforms=train_data, imputation_dict=imputation_dict)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used, drop_last=False)

        if val_data is not None:
            blah = next(iter(self.train_loader))
        else:
            blah = next(iter(self.test_loader))
        self.total_len = blah[0].shape[1]
        self.total_channels = blah[0].shape[2]

    def reload_model(self):
        self.epoch_list = [-1]
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path, "epoch_latest"), exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path, "epoch_best"), exist_ok=True)
        self.best_val_loss = 9999999
        if os.path.isfile(os.path.join(self.reload_ckpt_path,"epoch_best", "epoch_best.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path, "epoch_best", "epoch_best.pkl"), map_location=f"cuda:{self.gpu_list[0]}")
            best_epoch = state["iter"]
            print(f"Identified best iteration based on val: {best_epoch}")
            self.best_val_loss = state["msevalloss"]
        if os.path.isfile(os.path.join(self.reload_ckpt_path,f"epoch_{self.reload_epoch}", f"naomi_step_training_{self.reload_epoch}.pth")):
            state = torch.load(os.path.join(self.reload_ckpt_path ,f"epoch_{self.reload_epoch}", f"naomi_step_training_{self.reload_epoch}.pth"), 
                                map_location=f"cuda:{self.gpu_list[0]}")

            self.epoch_list.append(state["iter"])

            print(f"Reloading given epoch: {np.max(self.epoch_list)}")
            with open(os.path.join(self.reload_ckpt_path, "loss_log.txt"), 'a+') as f:
                f.write(f"Reloading newest epoch: {self.reload_epoch}\n")
            print(self.naomi_policy.load_state_dict(state['policy_state_dict'], strict=True))
            print(self.naomi_discrim.load_state_dict(state['discrim_state_dict'], strict=True))

            print(self.optimizer_policy.load_state_dict(state['optimizer_policy']))
            print(self.optimizer_discrim.load_state_dict(state['optimizer_discrim']))
        elif os.path.isfile(os.path.join(self.reload_ckpt_path,f"epoch_{self.reload_epoch}", f"epoch_{self.reload_epoch}.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path ,f"epoch_{self.reload_epoch}", f"epoch_{self.reload_epoch}.pkl"), 
                                 map_location=f"cuda:{self.gpu_list[0]}")
            self.epoch_list.append(state["iter"])

            print(f"Reloading given epoch: {np.max(self.epoch_list)}")
            with open(os.path.join(self.reload_ckpt_path, "loss_log.txt"), 'a+') as f:
                f.write(f"Reloading newest epoch: {self.reload_epoch}\n")
            print(self.naomi_policy.load_state_dict(state['policy_state_dict'], strict=True))
            print(self.naomi_discrim.load_state_dict(state['discrim_state_dict'], strict=True))

            print(self.optimizer_policy.load_state_dict(state['optimizer_policy']))
            print(self.optimizer_discrim.load_state_dict(state['optimizer_discrim']))
        elif os.path.exists(os.path.join(self.ckpt_path, "policy_step_pretrained.pth")) and \
            os.path.exists(os.path.join(self.ckpt_path, "discrim_step_pretrained.pth")):
            print(self.naomi_policy.load_state_dict(torch.load(os.path.join(self.ckpt_path, "policy_step_pretrained.pth"))))
            print(self.naomi_discrim.load_state_dict(torch.load(os.path.join(self.ckpt_path, "discrim_step_pretrained.pth"))))
        else:
            print(f"cannot reload epoch {self.reload_epoch}")

class pulseimpute_dataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, imputation_dict=None):
        'Initialization'
        self.waveforms = waveforms
        self.imputation_dict = imputation_dict

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.waveforms)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Load data and get label
        X = torch.clone(self.waveforms[idx, :, :])

        X_original = torch.clone(X)
        
        # mask = torch.isnan(y) # mask is where data is present
        # X = torch.cat([mask, X], 1) # back_gru has an extra dimension for mask
        
        y_dict = {"original": X_original,
                  "name": idx}

        if self.imputation_dict:
            y_dict["target_seq"] = self.imputation_dict["target_seq"][idx]
            mask = torch.isnan(y_dict["target_seq"])
            X = torch.cat([mask, X], -1)

        return X, y_dict

def create_missingness(waveforms, train_impute_wind, train_impute_prob, train_impute_extended, list_of_miss):
    mask = np.ones(waveforms.shape, dtype=np.float32)

    if list_of_miss:
        miss_idx = np.random.randint(len(list_of_miss))
        miss_vector = miss_tuple_to_vector(list_of_miss[miss_idx])
        miss_matrix = np.tile(miss_vector, (waveforms.shape[0], 1,1))
        mask = miss_matrix
        waveforms[np.where(miss_matrix == 0)] = 0
    elif not train_impute_extended:
        window = train_impute_wind
        probability = train_impute_prob 
        if window > 0 and probability > 0:
            rand = np.random.random_sample()
            # iterate over time
            for j in range(0, waveforms.shape[1], window):
                rand = np.random.random_sample()
                if rand <= probability:
                    mask[:,j:j+window, :] = 0
                    waveforms[:, j:j+window, :] = 0
    else:
        start_impute = np.random.randint(0, waveforms.shape[1]-train_impute_extended)
        mask[:,start_impute:start_impute+train_impute_extended, :] = 0
        waveforms[:,start_impute:start_impute+train_impute_extended, :] = 0
    waveforms = torch.cat([torch.Tensor(mask), waveforms], -1) 
    

    return waveforms, mask


def miss_tuple_to_vector(listoftuples):
    def onesorzeros_vector(miss_tuple):
        miss_tuple = literal_eval(miss_tuple)
        if miss_tuple[0] == 0:
            return np.zeros(miss_tuple[1])
        elif miss_tuple[0] == 1:
            return np.ones(miss_tuple[1])

    miss_vector = onesorzeros_vector(listoftuples[0])
    for i in range(1, len(listoftuples)):
        miss_vector = np.concatenate((miss_vector, onesorzeros_vector(listoftuples[i])))
    miss_vector = np.expand_dims(miss_vector, 1)
    return miss_vector
