import os
import multiprocessing
import torch
gpu_list = [2,3]
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from datetime import datetime
from .utils.load_data import load_data_mimic
from .utils.loss import l1_mpc_loss as combine_loss_func
from .utils.loss import l2_mpc_loss

from .utils.utils import make_attn_plot_stitch
from .utils.custom_convattn import TransformerEncoderLayer_CustomAttn,TransformerEncoder_CustomAttn
from tqdm import tqdm
import math

batch_size = 128
model_name = "notanh_mimic_mpc55_dcb_nobn_notlight" # this is the corrected version of the original wavenet with mode on and bounds off
path2_mimic_waveform = os.path.join(os.sep, "data", "mxu87", "mimic_waveform")
# path2_mimic_waveform = os.path.join(os.sep, "disk1", "mimic_waveform")
# path2_mimic_waveform = os.path.join(os.sep, "localscratch", "shared", "mxu87", "mimic_wav


class BertModel(torch.nn.Module):
    """Transformer language model.
    """
    def __init__(self, orig_dim=1, embed_dim=256, n_heads=4, max_len=1000, iter=3,kernel_size=21):
        super().__init__()
        self.iter = iter
        self.embed = DilatedStrideEmbed(orig_dim=orig_dim, embed_dim=embed_dim,kernel_size=kernel_size)
        
        #self.deconv = nn.ConvTranspose1d(embed_dim,orig_dim,kernel_size = kernel_size,stride=kernel_size)
        self.deconv = nn.ConvTranspose1d(embed_dim,orig_dim,kernel_size = kernel_size,stride=kernel_size,output_padding=13)
        
        q_k_func = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim*2, 
                             kernel_size=1, padding= 0)

        encoder_layer = TransformerEncoderLayer_CustomAttn(d_model=embed_dim, nhead = n_heads, activation="gelu", 
                                                          custom_attn=None, feedforward="fc", 
                                                          channel_pred=False, ssp=False, 
                                                          q_k_func= q_k_func,max_len=max_len)
        self.encoder = TransformerEncoder_CustomAttn(encoder_layer, num_layers=2)

        #self.mpc = BertMPCHead(orig_dim=orig_dim, embed_dim=embed_dim)
        
        #have to specify kernel size, block size
        self.kernel_size = 21 
        self.block_size = 7 #2

        self.pool = nn.AvgPool1d(kernel_size = self.kernel_size,stride = self.kernel_size)
        #assert self.kernel_size % self.block_size == 0
        self.pool2 = nn.AvgPool1d(kernel_size = self.block_size,stride = self.block_size)
        self.mean_outlier_layer = nn.Linear(orig_dim+1,orig_dim)

    def context_feats (self,in_series,test): #in_series: bs x length
        org_shape = in_series.shape[1]
        if (in_series.shape[1]%self.kernel_size != 0):
            in_series = torch.cat([in_series,torch.zeros(in_series.shape[0],self.kernel_size-in_series.shape[1]%self.kernel_size).to(in_series.device)],dim=1)
        feat1 = self.pool(in_series.unsqueeze(1))[:,0,:]
        feat1 = torch.repeat_interleave(feat1,self.kernel_size,dim=1)*self.kernel_size
        if (test):
            mask = (in_series != 0).int().float()
            den = self.pool(mask.unsqueeze(1))[:,0,:]
            den = (torch.repeat_interleave(den,self.kernel_size,dim=1)*self.kernel_size).clamp(min=1)
            out_feats = feat1/den
        else : 
            feat2 = self.pool2(in_series.unsqueeze(1))[:,0,:]
            feat2 = torch.repeat_interleave(feat2,self.block_size,dim=1)*self.block_size
            out_feats = (feat1 - feat2)/(self.kernel_size-self.block_size)
        return out_feats.unsqueeze(1)[:,:,:org_shape]                                                                                                                                                                         

    def forward(self, x, return_attn_weights=False, masktoken_bool=None,test=False): #shape of x: [batch_size, length, channels]
        embedding = self.embed(x) # shape [batch_size, embed_dim, length]
        embedding = embedding.permute(2,0,1) # shape [length, batch_size, embed_dim]
        if return_attn_weights:
            encoded, attn_weights_list = self.encoder(embedding, None, return_attn_weights=return_attn_weights) # shape [length, batch_size, embed_dim]
        else:
            encoded = self.encoder(embedding, None) # shape [length, batch_size, embed_dim]
        encoded = encoded.permute(1,2,0) # shape [batch_size, embed_dim, length]

        #mpc_projection = self.mpc(encoded) # shape [batch_size, embed_dim, length]
        #mpc_projection = mpc_projection.transpose(1,2) # shape [batch_size, length, embed_dim]
        mpc_projection = self.deconv(encoded).transpose(1,2)
        
        #for deep MVI: add local context computation
        local_feats = self.context_feats(x[:,:,0],test).transpose(1,2) #before transpose shape is [bs,1,length]
        
        #fixing the size that changed after deconv and repeat_interleave
        #temp_feats = torch.zeros(local_feats.shape[0],max(local_feats.shape[1],mpc_projection.shape[1]),local_feats.shape[2]).to(local_feats.device)
        #temp_mpc = torch.zeros(mpc_projection.shape[0],max(local_feats.shape[1],mpc_projection.shape[1]),mpc_projection.shape[2]).to(mpc_projection.device)
        #temp_feats[:,0:local_feats.shape[1],:] = local_feats
        #temp_mpc[:,0:mpc_projection.shape[1],:] = mpc_projection
        
        mpc_projection = mpc_projection[:,0:local_feats.shape[1],:]
        feats = torch.cat([mpc_projection,local_feats],dim=2) #shape [bs,length,1+org_dim]
        #feats = torch.cat([temp_mpc,temp_feats],dim=2) #shape [bs,length,1+org_dim]
        mean = self.mean_outlier_layer(feats) #[:,:,0]
        if return_attn_weights:
            return mean, attn_weights_list #mpc_projection, attn_weights_list
        else:
            return mean #mpc_projection


class MaskTokenAdder(nn.Module):

    def __init__(self, d_model, orig_dim):
        super().__init__()
        self.d_model = d_model
        self.orig_dim = orig_dim

        self.mask_token = nn.Parameter(torch.randn(d_model))

    def forward(self, x, masktoken_bool):
        # 1s at normal vals 0s at mask vals
        added_masks = torch.where(masktoken_bool.repeat(1,self.d_model // self.orig_dim,1) , x, 
                        self.mask_token.unsqueeze(0).unsqueeze(-1).repeat(x.shape[0], 1, x.shape[2]))
        return added_masks
# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/position.py
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model,kernel_size=20, max_len=512):
        super().__init__()
        max_len = int((max_len - kernel_size -2)/kernel_size + 1)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].transpose(1,2)
class FC_adapter(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.fc = nn.Sequential(
            torch.nn.Linear(dim1, dim2),
            nn.GELU(),
            torch.nn.Linear(dim2, dim3),
            nn.GELU(),
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class PredictionHead(torch.nn.Module):
    def __init__(self, embed_dim=32, classes=2):
        super().__init__()
        self.pooler = torch.nn.Linear(embed_dim, embed_dim)
        with torch.no_grad():
            self.pooler.bias.zero_()
        self.activation = torch.tanh
        self.origchannel_logits = torch.nn.Linear(embed_dim, classes)

    def forward(self, embedded_states):
        # embedded_states: [batch size, embed_dim]
        # sequence_index: index of the token to pool.
        pooled = self.pooler(embedded_states)
        pooled = self.activation(pooled)

        channel_logits = self.origchannel_logits(pooled)
        return channel_logits

class DilatedStrideEmbed(torch.nn.Module):
    def __init__(self, orig_dim=12,embed_dim=32,kernel_size=11):
        super().__init__()
        # output_size=(w+2*pad-(d(k-1)+1))/s+1
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=orig_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=kernel_size, padding=1, dilation=1)
        )
    def forward(self, x):
        # x stores integers and has shape [batch_size, length, channels]        
        x = x.permute(0,2,1)
        x1 = self.embedding(x)
        return x1

class BertMPCHead(torch.nn.Module):
    """Masked PC head for Bert
    The model structure of MPC is essentially the encoder part of Transformer based
    model plus a single fully-connected projection layer

    Arguments:
        hidden_size: hidden size
        output_size: output size
    """
    def __init__(self, orig_dim=12, embed_dim = 32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=orig_dim, kernel_size=11, stride=1, padding=5*1, dilation=1))

    def forward(self, encoded_states):
        original = self.projection(encoded_states)
        return original


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

from torch.optim.lr_scheduler import LambdaLR
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_ep