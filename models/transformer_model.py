import os
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm
import csv
from ast import literal_eval
import time

from utils.loss_mask import mse_mask_loss
from .transformer.utils.loss import l2_mpc_loss


class transformer():
    def __init__(self,modelname, data_name, train_data=None, val_data=None, imputation_dict= None, annotate="",annotate_test="",
                 bs= 64, gpus=[0,1], train_time=999, # in hours
                 train_impute_wind=None, train_impute_prob=None, train_impute_extended=None,
                 train_realecg=False, train_realppg=False,
                 max_len=None, iter_save=None,
                 reload_epoch=-1,
                 convertolong=None, reload_epoch_long=None,
                 deepmvi=False):

        if deepmvi:
            from .transformer.utils.custom_convattn_deepmvi import TransformerEncoderLayer_CustomAttn, TransformerEncoder_CustomAttn
        else:
            from .transformer.utils.custom_convattn import TransformerEncoderLayer_CustomAttn, TransformerEncoder_CustomAttn
        outpath = "out/"
        self.iter_save = iter_save
        self.train_time = train_time

        if convertolong:
            from .transformer.utils.custom_convlongattn import Custom_LongformerSelfAttn
            print("loading tvm")
            from longformer.diagonaled_mm_tvm import diagonaled_mm

        if "masktoken" in modelname:
            self.masktoken = True
        else:
            self.masktoken=False
        self.data_name = data_name
        self.bs = bs
        self.gpu_list = gpus
        self.train_realppg = train_realppg
        self.train_realecg = train_realecg
        self.train_impute_extended=train_impute_extended
        self.train_impute_wind = train_impute_wind
        self.train_impute_prob = train_impute_prob
        self.annotate_test = annotate_test

        if reload_epoch == -1:
            self.reload_epoch = "latest"
        else:
            self.reload_epoch = reload_epoch

        self.data_loader_setup(train_data, val_data, imputation_dict=imputation_dict)

        # cannot get relative import working for the life of me
        model_module = __import__(f'models.transformer.{modelname}', fromlist=[""])
        model_module_class = getattr(model_module, "BertModel")
        print(self.gpu_list)
        if len(self.gpu_list) == 1:
            torch.cuda.set_device(self.gpu_list[0])


        self.model =  nn.DataParallel(model_module_class(orig_dim=self.total_channels, max_len=max_len), device_ids=self.gpu_list)

        self.model.to(torch.device(f"cuda:{self.gpu_list[0]}"))
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total params is {}'.format(total_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)

        self.ckpt_path = os.path.join(outpath, data_name+annotate_test, modelname+annotate)
        self.reload_ckpt_path = os.path.join(outpath, data_name, modelname+annotate)
        self.reload_model()

        if convertolong:
            print("converting transformer to longformer")
            class configforlong():
                def __init__(self, num_heads, hidden_size):
                    self.num_attention_heads = num_heads
                    self.hidden_size = hidden_size
                    self.autoregressive = False
                    self.attention_probs_dropout_prob = .1
                    self.attention_window = convertolong["attention_window"]
                    self.attention_dilation = convertolong["attention_dilation"]
                    self.attention_mode = "tvm" #  ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
            config = configforlong(self.model.module.encoder.layers[0].self_attn.num_heads,
                                self.model.module.encoder.layers[0].self_attn.embed_dim)
            if "van" in modelname:
                self.model.module.pos_embed = PositionalEmbedding(self.model.module.encoder.layers[0].self_attn.in_proj_weight.shape[1], 30000).to(torch.device(f"cuda:{self.gpu_list[0]}"))
                encoder_layer = TransformerEncoderLayer_CustomAttn(d_model=self.model.module.encoder.layers[0].self_attn.embed_dim, 
                                                          nhead = self.model.module.encoder.layers[0].self_attn.num_heads, activation="gelu", 
                                                          custom_attn=None, feedforward="fc", 
                                                          channel_pred=False, ssp=False)
                encoder = TransformerEncoder_CustomAttn(encoder_layer, num_layers=2)

            for idx, encoderlayer in enumerate(self.model.module.encoder.layers):
                if "van" in modelname:
                    long_self_attn = Custom_LongformerSelfAttn(config, idx)
                    q_func  = nn.Linear(int(encoderlayer.self_attn.in_proj_weight.shape[0]/3), encoderlayer.self_attn.in_proj_weight.shape[1]) # 768, 256
                    q_func.weight = nn.Parameter(encoderlayer.self_attn.in_proj_weight[:int(encoderlayer.self_attn.in_proj_weight.shape[0]/3),:])
                    q_func.bias = nn.Parameter(encoderlayer.self_attn.in_proj_bias[:int(encoderlayer.self_attn.in_proj_weight.shape[0]/3)])

                    k_func  = nn.Linear(int(encoderlayer.self_attn.in_proj_weight.shape[0]/3), encoderlayer.self_attn.in_proj_weight.shape[1]) # 768, 256
                    k_func.weight = nn.Parameter(encoderlayer.self_attn.in_proj_weight[int(encoderlayer.self_attn.in_proj_weight.shape[0]/3):int(2*encoderlayer.self_attn.in_proj_weight.shape[0]/3),:])
                    k_func.bias = nn.Parameter(encoderlayer.self_attn.in_proj_bias[int(encoderlayer.self_attn.in_proj_weight.shape[0]/3):int(2*encoderlayer.self_attn.in_proj_weight.shape[0]/3)])

                    v_func  = nn.Linear(int(encoderlayer.self_attn.in_proj_weight.shape[0]/3), encoderlayer.self_attn.in_proj_weight.shape[1]) # 768, 256
                    v_func.weight = nn.Parameter(encoderlayer.self_attn.in_proj_weight[int(2*encoderlayer.self_attn.in_proj_weight.shape[0]/3):,:])
                    v_func.bias = nn.Parameter(encoderlayer.self_attn.in_proj_bias[int(2*encoderlayer.self_attn.in_proj_weight.shape[0]/3):])
                    
                    long_self_attn.query = q_func
                    long_self_attn.key = k_func
                    long_self_attn.value = v_func                    
                    encoder.layers[idx].self_attn = long_self_attn
                else:
                    long_self_attn = Custom_LongformerSelfAttn(config, idx, q_k_func=encoderlayer.self_attn.q_k_conv)
                    long_self_attn.value = encoderlayer.self_attn.v_linear
                    encoderlayer.self_attn  = long_self_attn.to(torch.device(f"cuda:{self.gpu_list[0]}"))
            if "van" in modelname:
                self.model.module.encoder = encoder.to(torch.device(f"cuda:{self.gpu_list[0]}"))
            
            self.model.module.embed = nn.Sequential(self.model.module.embed, PermuteModule()).to(torch.device(f"cuda:{self.gpu_list[0]}"))
            if deepmvi:
                self.model.module.deconv = nn.Sequential(PermuteModule(2,1,0), self.model.module.deconv)
            else:
                self.model.module.mpc = nn.Sequential(PermuteModule(2,1,0), self.model.module.mpc)
            print("finished converting")

            if reload_epoch_long: # this is where we are loading a longformer epoch
                state = torch.load(os.path.join(self.reload_ckpt_path ,f"epoch_{reload_epoch_long}", f"epoch_{reload_epoch_long}.pkl"), 
                                map_location=f"cuda:{self.gpu_list[0]}")
                self.epoch_list.append(state["epoch"])

                print(f"Reloading given epoch: {np.max(self.epoch_list)}")
                with open(os.path.join(self.reload_ckpt_path, "loss_log.txt"), 'a+') as f:
                    f.write(f"Reloading newest epoch: {self.reload_epoch}\n")
                print(self.model.load_state_dict(state['state_dict'], strict=True))


    def data_loader_setup(self, train_data, val_data=None, imputation_dict=None):
        num_threads_used = multiprocessing.cpu_count() 
        # num_threads_used = multiprocessing.cpu_count() // 8 can set this as some function of threads per gpu
        print(f"Num Threads Used: {num_threads_used}")
        torch.set_num_threads(num_threads_used)
        os.environ["MP_NUM_THREADS"]=str(num_threads_used)
        os.environ["OPENBLAS_NUM_THREADS"]=str(num_threads_used)
        os.environ["MKL_NUM_THREADS"]=str(num_threads_used)
        os.environ["VECLIB_MAXIMUM_THREADS"]=str(num_threads_used)
        os.environ["NUMEXPR_NUM_THREADS"]=str(num_threads_used)

        if val_data is not None:
            train_dataset = mpr_dataset(waveforms=train_data, 
                                        train_impute_wind=self.train_impute_wind, train_impute_prob=self.train_impute_prob, 
                                        train_impute_extended=self.train_impute_extended,
                                        train_realppg=self.train_realppg,
                                        train_realecg=self.train_realecg, type="train")
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=num_threads_used)
            temp = next(iter(self.train_loader))

            val_dataset = mpr_dataset(waveforms=val_data, 
                                        train_impute_wind=self.train_impute_wind, train_impute_prob=self.train_impute_prob, 
                                        train_impute_extended=self.train_impute_extended,
                                        train_realppg=self.train_realppg,
                                        train_realecg=self.train_realecg, type="val")
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
        else:
            test_dataset = mpr_dataset(waveforms=train_data, imputation_dict=imputation_dict,
                                        train_impute_wind=None, train_impute_prob=None, 
                                        train_impute_extended=None,
                                        train_realppg=False,
                                        train_realecg=False, type="test")
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
            temp = next(iter(self.test_loader))

        self.total_channels = temp[0].shape[2]

    def reload_model(self):
        self.epoch_list = [-1]
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path, "epoch_latest"), exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path, "epoch_best"), exist_ok=True)
        self.best_val_loss = 9999999
        if os.path.isfile(os.path.join(self.reload_ckpt_path,"epoch_best", "epoch_best.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path, "epoch_best", "epoch_best.pkl"), map_location=f"cuda:{self.gpu_list[0]}")
            best_epoch = state["epoch"]
            print(f"Identified best epoch: {best_epoch}")
            self.best_val_loss = state["l2valloss"].cpu()
        if os.path.isfile(os.path.join(self.reload_ckpt_path,f"epoch_{self.reload_epoch}", f"epoch_{self.reload_epoch}.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path ,f"epoch_{self.reload_epoch}", f"epoch_{self.reload_epoch}.pkl"), 
                                map_location=f"cuda:{self.gpu_list[0]}")
            self.epoch_list.append(state["epoch"])

            print(f"Reloading given epoch: {np.max(self.epoch_list)}")
            with open(os.path.join(self.reload_ckpt_path, "loss_log.txt"), 'a+') as f:
                f.write(f"Reloading newest epoch: {self.reload_epoch}\n")
            print(self.model.load_state_dict(state['state_dict'], strict=True))
            print(self.optimizer.load_state_dict(state['optimizer']))
        else:
            print(f"cannot reload epoch {self.reload_epoch}")
                

    def testimp(self):
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")

        print(f'{dt_string} | Start')

        with torch.no_grad():
            self.model.eval()
            total_test_mse_loss = 0
            total_missing_total = 0

            flag=True
            residuals_all = []
            for local_batch, local_label_dict in tqdm(self.test_loader, desc="Testing", leave=False):  
                # 1s at normal vals 0s at mask vals
                if self.masktoken:
                    mask = torch.isnan(local_label_dict["target_seq"])
                    mpc_projection = self.model(local_batch, masktoken_bool=mask)
                else:
                    mpc_projection = self.model(local_batch)
                mse_loss,missing_total, residuals = mse_mask_loss(mpc_projection.to(torch.device(f"cuda:{self.gpu_list[0]}")), 
                                            local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")),
                                            residuals=True)
                residuals_all.append(residuals)
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
        total_test_mpcl2_std = torch.std(torch.cat(residuals_all))

        epoch_check_path = os.path.join(self.ckpt_path, "epoch_" +  str(self.reload_epoch))
        os.makedirs(epoch_check_path, exist_ok=True)

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | MSE:{total_test_mse_loss:.10f} | Std SE: {total_test_mpcl2_std:.10f}  \n')
        with open(os.path.join(epoch_check_path, "loss_log.txt"), 'a+') as f:
            f.write(f'{dt_string} | MSE:{total_test_mse_loss:.10f} | Std SE: {total_test_mpcl2_std:.10f}  \n')

        np.save(os.path.join(epoch_check_path, "imputation.npy"), imputation_cat)


    def train(self):

        writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "tb"))

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | Start')

        start = time.time()

        iter_idx = 0
        for epoch in range(np.max(self.epoch_list)+1, 1000):
            total_train_mpcl2_loss = 0
            total_missing_total = 0

            if self.iter_save:
                
                for local_batch, local_label_dict in tqdm(self.train_loader, desc="Training", leave=False):
                    iter_idx += 1
                    self.optimizer.zero_grad()
                    if self.masktoken:
                        mask = torch.isnan(local_label_dict["target_seq"])
                        mpc_projection = self.model(local_batch, masktoken_bool=mask)
                    else:
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

                            flag=True
                            val_iter_idx = 0
                            for local_batch, local_label_dict in tqdm(self.val_loader, desc="Validating", leave=False): 
                                val_iter_idx += 1
                                if self.masktoken:
                                    mask = torch.isnan(local_label_dict["target_seq"])
                                    mpc_projection = self.model(local_batch, masktoken_bool=mask)
                                else:
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
                            except:
                                pass
                            torch.save(state, os.path.join(self.ckpt_path, "epoch_best", "epoch_best.pkl"))
                            self.best_val_loss = total_val_mpcl2_loss
                            best_epoch = iter_idx


                        total_train_mpcl2_loss = 0
                        total_missing_total = 0


            else:
                for local_batch, local_label_dict in tqdm(self.train_loader, desc="Training", leave=False):
                    end = time.time()
                    if (end-start) / 60 / 60 >= self.train_time:
                        import sys; sys.exit()

                    self.optimizer.zero_grad()
                    if self.masktoken:
                        mask = torch.isnan(local_label_dict["target_seq"])
                        mpc_projection = self.model(local_batch, masktoken_bool=mask)
                    else:
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
                        if self.masktoken:
                            mask = torch.isnan(local_label_dict["target_seq"])
                            mpc_projection = self.model(local_batch, masktoken_bool=mask)
                        else:
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
                    best_epoch = epoch

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




class mpr_dataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, train_impute_wind=21, train_impute_prob=.15, train_impute_extended=None,
                 randimputeallchannel=False, imputation_dict=None,
                train_realppg=None,
                train_realecg=False,  type=None):
        'Initialization'
        self.waveforms = waveforms
        self.train_impute_wind = train_impute_wind
        self.train_impute_prob = train_impute_prob
        self.train_impute_extended = train_impute_extended
        self.randimputeallchannel = randimputeallchannel
        self.imputation_dict = imputation_dict
        self.train_realppg = train_realppg
        self.train_realecg = train_realecg
        if train_realecg or train_realppg:
            if train_realppg:
                tuples_path = os.path.join("data", f"missing_ppg_{type}.csv")
            elif train_realecg:
                tuples_path = os.path.join("data", f"missing_ecg_{type}.csv")

            with open(tuples_path, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                self.list_of_miss = list(csv_reader)

        assert train_impute_extended and not (train_impute_wind and train_impute_prob) or \
            not train_impute_extended and (train_impute_wind and train_impute_prob) or \
            not train_impute_extended and not train_impute_wind and not train_impute_prob or \
                train_realecg or \
                train_realppg


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.waveforms)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Load data and get label
        X = torch.clone(self.waveforms[idx, :, :])

        X_original = torch.clone(X)
        y = np.empty(X.shape, dtype=np.float32)
        y[:] = np.nan
        y = torch.from_numpy(y)

        # lets randomly mask!
        # iterate over channels
        if not self.imputation_dict:
            if self.train_realecg or self.train_realppg:
                miss_idx = np.random.randint(len(self.list_of_miss))
                miss_vector = miss_tuple_to_vector(self.list_of_miss[miss_idx])
                y[np.where(miss_vector == 0)] = X[np.where(miss_vector == 0)]
                rand = np.random.random_sample()
                if rand <= 0.10: 
                    pass
                # Randomly Replace
                elif rand <= 0.20:
                    # 50 Hz for Powerline Interference -> 1/50 sec period -> 2 centisec per -> 2 = 2pi/B -> B = 2
                    start_idx = 0
                    for miss_tuple in self.list_of_miss[miss_idx]:
                        miss_tuple = literal_eval(miss_tuple)
                        if miss_tuple[0]==0:
                            X[start_idx:miss_tuple[1]+start_idx]+= np.expand_dims(.1*np.cos(2*np.arange(0,miss_tuple[1])), axis = 1) 
                        start_idx += miss_tuple[1]
                # Set to 0
                else:
                    X[np.where(miss_vector == 0)] = 0

            elif self.train_impute_extended:
                start_impute = np.random.randint(0, X.shape[0]-self.train_impute_extended)
                y[start_impute:start_impute+self.train_impute_extended, :] = X[start_impute:start_impute+self.train_impute_extended, :]
                rand = np.random.random_sample()
                if rand <= 0.10: 
                    pass
                elif rand <= 0.20:
                    X[start_impute:start_impute+self.train_impute_extended, :]+= np.expand_dims(.1*np.cos(2*np.arange(0,self.train_impute_extended)), axis = 1) 
                else:
                    X[start_impute:start_impute+self.train_impute_extended, :] = 0

            else:
                window = self.train_impute_wind
                probability = self.train_impute_prob 
                # iterate over time
                for j in range(0, X.shape[0], window):
                    rand = np.random.random_sample()
                    if rand <= probability:
                        if X.shape[0]-j <  window:
                            incr = X.shape[0]-j
                        else:
                            incr = window
                        y[j:j+incr, :] = X[j:j+incr, :]
                        rand = np.random.random_sample()
                        if rand <= 0.10: 
                            continue
                        elif rand <= 0.20:
                            X[j:j+incr, :] += np.expand_dims(.1*np.cos(2*np.arange(0,incr)), axis = 1) 
                        else:
                            X[j:j+incr, :] = 0
    
        y_dict = {"target_seq": y,
                  "original": X_original,
                  "name": idx}

        if self.imputation_dict:
            y_dict["target_seq"] = self.imputation_dict["target_seq"][idx]

        return X, y_dict


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

class PermuteModule(torch.nn.Module):
    def __init__(self, dim1=2, dim2=1, dim3=0):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
    def forward(self, x):
        return x.permute(self.dim1, self.dim2, self.dim3)

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].permute(1,2,0)