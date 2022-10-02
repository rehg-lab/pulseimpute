import torch
import torch.nn as nn
import numpy as np
import os
import csv
from ast import literal_eval

def l2_mpc_loss(logits , target, residuals=False):
    logits_mpc = torch.clone(logits)
    target_mpc = torch.clone(target)
    logits_mpc[torch.isnan(target)] = 0
    target_mpc[torch.isnan(target)] = 0
    difference = torch.square(logits_mpc - target_mpc)
    l2_loss = torch.sum(difference)
    
    if residuals:
        return l2_loss, difference[~torch.isnan(target)]
    else:
        return l2_loss

def reload_model(model, optimizer, reload_epoch, reload_ckpt_path, gpu=0):
    epoch_list = [-1]
    os.makedirs(reload_ckpt_path, exist_ok=True)
    os.makedirs(os.path.join(reload_ckpt_path, "epoch_latest"), exist_ok=True)
    os.makedirs(os.path.join(reload_ckpt_path, "epoch_best"), exist_ok=True)

    best_val_loss = 9999999
    if os.path.isfile(os.path.join(reload_ckpt_path,"epoch_best", "epoch_best.pkl")):
        state = torch.load(os.path.join(reload_ckpt_path, "epoch_best", "epoch_best.pkl"), map_location=f"cuda:{gpu}")
        best_epoch = state["epoch"]
        print(f"Identified best epoch: {best_epoch}")
        best_val_loss = state["l2valloss"].cpu()
        
    if os.path.isfile(os.path.join(reload_ckpt_path,f"epoch_{reload_epoch}", f"epoch_{reload_epoch}.pkl")):

        state = torch.load(os.path.join(reload_ckpt_path ,f"epoch_{reload_epoch}", f"epoch_{reload_epoch}.pkl"), 
                            map_location=f"cuda:{gpu}")
        epoch_list.append(state["epoch"])

        print(f"Reloading given epoch: {np.max(epoch_list)}")
        with open(os.path.join(reload_ckpt_path, "loss_log.txt"), 'a+') as f:
            f.write(f"Reloading newest epoch: {reload_epoch}\n")
        print(model.load_state_dict(state['state_dict'], strict=True))
        print(optimizer.load_state_dict(state['optimizer']))
    else:
        print(f"cannot reload epoch {reload_epoch}")

    return epoch_list, best_val_loss
                


def transformer_to_longformer(transformer_model, modelname, converttolong_dict, gpu=0):

    from .custom_convlongattn import Custom_LongformerSelfAttn
    print("Loading tvm for longformer")
    from longformer.diagonaled_mm_tvm import diagonaled_mm

    if "deepmvi" not in modelname:
        from .custom_convattn import TransformerEncoderLayer_CustomAttn, TransformerEncoder_CustomAttn     
    else:
        from .custom_convattn_deepmvi import TransformerEncoderLayer_CustomAttn, TransformerEncoder_CustomAttn

    print("Converting Transformer to Longformer")
    class configforlong():
        def __init__(self, num_heads, hidden_size):
            self.num_attention_heads = num_heads
            self.hidden_size = hidden_size
            self.autoregressive = False
            self.attention_probs_dropout_prob = .1
            self.attention_window = converttolong_dict["attention_window"]
            self.attention_dilation = converttolong_dict["attention_dilation"]
            self.attention_mode = "tvm"

    config = configforlong(transformer_model.module.encoder.layers[0].self_attn.num_heads,
                        transformer_model.module.encoder.layers[0].self_attn.embed_dim)

    if "van" in modelname:
        transformer_model.module.pos_embed = PositionalEmbedding(transformer_model.module.encoder.layers[0].self_attn.in_proj_weight.shape[1], 30000).to(torch.device(f"cuda:{gpu}"))
        encoder_layer = TransformerEncoderLayer_CustomAttn(d_model=transformer_model.module.encoder.layers[0].self_attn.embed_dim, 
                                                    nhead = transformer_model.module.encoder.layers[0].self_attn.num_heads, activation="gelu", 
                                                    custom_attn=None, feedforward="fc", 
                                                    channel_pred=False, ssp=False)
        encoder = TransformerEncoder_CustomAttn(encoder_layer, num_layers=2)

    for idx, encoderlayer in enumerate(transformer_model.module.encoder.layers):
        if "van" not in modelname:
            long_self_attn = Custom_LongformerSelfAttn(config, idx, q_k_func=encoderlayer.self_attn.q_k_conv)
            long_self_attn.value = encoderlayer.self_attn.v_linear
            encoderlayer.self_attn  = long_self_attn.to(torch.device(f"cuda:{gpu}"))
        else:
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
            
    if "van" in modelname:
        transformer_model.module.encoder = encoder.to(torch.device(f"cuda:{gpu}"))
    
    transformer_model.module.embed = nn.Sequential(transformer_model.module.embed, PermuteModule()).to(torch.device(f"cuda:{gpu}"))

    if "deepmvi" not in modelname:
        transformer_model.module.mpc = nn.Sequential(PermuteModule(2,1,0), transformer_model.module.mpc)
    else:
        transformer_model.module.deconv = nn.Sequential(PermuteModule(2,1,0), transformer_model.module.deconv)

    print("Finished converting")


class PermuteModule(torch.nn.Module):
    def __init__(self, dim1=2, dim2=1, dim3=0):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
    def forward(self, x):
        return x.permute(self.dim1, self.dim2, self.dim3)

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


class mpc_dataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, train_impute_wind=None, train_impute_prob=None, train_impute_extended=None,
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
                tuples_path = os.path.join("data","missingness_patterns", f"missing_ppg_{type}.csv")
            elif train_realecg:
                tuples_path = os.path.join("data","missingness_patterns", f"missing_ecg_{type}.csv")

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