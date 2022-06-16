import os
import multiprocessing
import torch
gpu_list = [2,3]
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from datetime import datetime
from .utils.load_data import load_data_mimic

from .utils.utils import make_impute_plot, make_attn_plot_stitch, make_confusion_matrix_plot
import re
from tqdm import tqdm
from prettytable import PrettyTable
import torch.nn.functional as F
import math

batch_size = 128


class LSTMModel(torch.nn.Module): 
    """Simple LSTM model
    """
    def __init__(self, orig_dim=1, embed_dim=256, n_layers=4, max_len=1000):
        super().__init__()
        self.lstm = nn.LSTM(orig_dim,embed_dim,n_layers,batch_first=True) 
        self.fc = nn.Linear(embed_dim,orig_dim)


    def forward(self, x): #shape: [batch_size, length, orig_dim]
        lstm_out,_ = self.lstm(x)
        fc_out = self.fc(lstm_out)

        return fc_out

"""
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
#     print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
"""


"""
if __name__=='__main__':
    num_threads_used = multiprocessing.cpu_count() // torch.cuda.device_count() * len(gpu_list)
    # num_threads_used = multiprocessing.cpu_count() 
    print(f"Num Threads Used: {num_threads_used}")
    torch.set_num_threads(num_threads_used)
    os.environ["MP_NUM_THREADS"]=str(num_threads_used)
    os.environ["OPENBLAS_NUM_THREADS"]=str(num_threads_used)
    os.environ["MKL_NUM_THREADS"]=str(num_threads_used)
    os.environ["VECLIB_MAXIMUM_THREADS"]=str(num_threads_used)
    os.environ["NUMEXPR_NUM_THREADS"]=str(num_threads_used)
    
    if len(gpu_list) == 1:
        torch.cuda.set_device(gpu_list[0])

    model = LSTMModel()

    model = nn.DataParallel(model, device_ids=gpu_list)
    model.to(torch.device(f"cuda:{gpu_list[0]}"))

    count_parameters(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6*batch_size)

    checkpoint_loc = os.path.join("checkpoints_lmao", model_name)
    epoch_list = [-1]

    try:
        os.mkdir(checkpoint_loc)
        os.mkdir(os.path.join(checkpoint_loc, "latest"))
        os.mkdir(os.path.join(checkpoint_loc, "best"))
    except:
        print("Folder already exists watch out! Continue to reload newest epoch?")
        try:
            state = torch.load(os.path.join(checkpoint_loc,"latest", "latest_epoch.pkl"), map_location=f"cuda:{gpu_list[0]}")
            epoch_list.append(state["epoch"])

            print(f"Reloading newest epoch: {np.max(epoch_list)}")
            with open(os.path.join(checkpoint_loc, "loss_log.txt"), 'a+') as f:
                f.write(f"Reloading newest epoch: {np.max(epoch_list)}\n")
            print(model.load_state_dict(state['state_dict'], strict=True))
            print(optimizer.load_state_dict(state['optimizer']))
            
            state = torch.load(os.path.join(checkpoint_loc,"best", "best_epoch.pkl"), map_location=f"cuda:{gpu_list[0]}")
            best_epoch = state["epoch"]
            print(f"Identified best epoch: {best_epoch}")
        except:
            pass

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50000, num_training_steps=10e6, last_epoch=np.max(epoch_list))
    imp_wind = 55
    train_loader, test_loader = load_data_mimic(window=imp_wind, batch_size=batch_size,
        path = path2_mimic_waveform, ssp=False, chanpred=True, noV=True, mode=True,bounds=1,
        noise=True, num_workers = num_threads_used, npy=True, reproduce=True)

    writer = SummaryWriter(log_dir=os.path.join(checkpoint_loc, "tb"))

    min_total_test_loss = 9999999999999999
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
    print(f'{dt_string} | Start')

    for epoch in range(np.max(epoch_list)+1, 100000000):
        
        total_train_mpcl2_loss = 0
        for local_batch, local_label_dict in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            model_out = model(local_batch)
            
            l2loss = l2_mpc_loss(model_out.to(torch.device(f"cuda:{gpu_list[0]}"),local_label_dict["target_seq"].to(torch.device(f"cuda:{gpu_list[0]}"))
            l2_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_train_mpcl2_loss += l2_loss.item()

        total_train_mpcl2_loss /=  len(train_loader.dataset)
        writer.add_scalar('Masked Predictive Coding L1 Loss/Train', total_train_mpcl2_loss, epoch)

        with torch.no_grad():
            model.eval()
            total_test_mpcl1_loss = 0
            total_test_mpcl2_loss = 0

            iter = 1
            for local_batch, local_label_dict in tqdm(test_loader, desc="Testing", leave=False):     
                mpc_projection = model(local_batch)

                mpcl1_loss = combine_loss_func(
                                                mpc_projection.to(torch.device(f"cuda:{gpu_list[0]}")), 
                                                local_label_dict["target_seq"].to(torch.device(f"cuda:{gpu_list[0]}"))
                                                )
                real_mpcl2_loss = l2_mpc_loss(
                                                mpc_projection.to(torch.device(f"cuda:{gpu_list[0]}")), 
                                                local_label_dict["target_seq"].to(torch.device(f"cuda:{gpu_list[0]}"))
                                                )
                total_test_mpcl1_loss += mpcl1_loss.item()
                total_test_mpcl2_loss += real_mpcl2_loss.item()

                if iter == len(test_loader):
                    mpc_projection_stitch = torch.zeros(local_batch.shape).to(torch.device(f"cuda:{gpu_list[0]}"))
                    attn_weights_stitch = []
                    local_batch = torch.clone(local_label_dict["original"])
                    freq = np.ceil(1000/imp_wind/10) #calculating freq of # imp wind until attn weights calc s.t. there are 10 attn positions

                    for start in range(0,1000,imp_wind):
                        local_batch[:, start:start+imp_wind, :] = 0.0

                        if start % (imp_wind*freq) == 0:
                            mpc_projection, attn_weights = model(local_batch, return_attn_weights=True)
                            attn_weights_stitch.append(attn_weights[0][-1, :, :].unsqueeze(0).cpu().detach().numpy()) #fucking annoying channel thing
                        else:
                            mpc_projection = model(local_batch)

                        mpc_projection_stitch[:,start:start+imp_wind,:] = mpc_projection[:,start:start+imp_wind,:]
                        local_batch = torch.clone(local_label_dict["original"])
                iter += 1
            model.train()
        total_test_mpcl2_loss /=  len(test_loader.dataset) - 1
        total_test_mpcl1_loss /=  len(test_loader.dataset) - 1
        writer.add_scalar('Masked Predictive Coding Loss/Test', total_test_mpcl2_loss, epoch)
        writer.add_scalar('Masked Predictive Coding Loss L1/Test', total_test_mpcl1_loss, epoch)

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f'{dt_string} | Epoch: {epoch} \nTrain MPC L1 Loss: {total_train_mpcl2_loss:.3f} \nTest MPC L1 Loss:{total_test_mpcl1_loss:.3f} \nTest MPC L2 Loss:{total_test_mpcl2_loss:.3f} \n')
        with open(os.path.join(checkpoint_loc, "loss_log.txt"), 'a+') as f:
            f.write(f'{dt_string} | Epoch: {epoch} \nTrain MPC L1 Loss: {total_train_mpcl2_loss:.3f} \nTest MPC L1 Loss:{total_test_mpcl1_loss:.3f} \nTest MPC L2 Loss:{total_test_mpcl2_loss:.3f} \n')

        if epoch != 0:
            os.remove(os.path.join(checkpoint_loc, "latest", "latest_epoch.pkl"))

        state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
        torch.save(state, os.path.join(checkpoint_loc, "latest", "latest_epoch.pkl"))
        if total_test_mpcl2_loss <= min_total_test_loss:
            if epoch != 0:
                os.remove(os.path.join(checkpoint_loc, "best", "best_epoch.pkl"))
            torch.save(state, os.path.join(checkpoint_loc, "best", "best_epoch.pkl"))
            min_total_test_loss = total_test_mpcl2_loss
            best_epoch = epoch

        if epoch % 1 == 0:
            epoch_check_path = os.path.join(checkpoint_loc, "epoch_" +  str(epoch))
            if epoch % 5 == 0:
                try:
                    os.mkdir(epoch_check_path)
                    torch.save(state, os.path.join(epoch_check_path, "epoch_" +  str(epoch) + ".pkl"))
                except:
                    pass
            
    writer.close()
"""
