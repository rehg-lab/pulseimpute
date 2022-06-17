import os
import multiprocessing
import torch
gpu_list = [2,3]
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from datetime import datetime
from .utils.load_data import load_data_mimic
from .utils.loss import l1_mpc_loss as combine_loss_func, class_total_correct
from .utils.loss import l2_mpc_loss

from .utils.utils import make_impute_plot, make_attn_plot_stitch, make_confusion_matrix_plot
from .utils.custom_convattn import TransformerEncoderLayer_CustomAttn, NearestNeighborsEmbed, TransformerEncoder_CustomAttn
import re
from tqdm import tqdm

import torch.nn.functional as F
import math

batch_size = 128
model_name = "notanh_mimic_mpc55_dcb_nobn_notlight" # this is the corrected version of the original wavenet with mode on and bounds off
path2_mimic_waveform = os.path.join(os.sep, "data", "mxu87", "mimic_waveform")
# path2_mimic_waveform = os.path.join(os.sep, "disk1", "mimic_waveform")
# path2_mimic_waveform = os.path.join(os.sep, "localscratch", "shared", "mxu87", "mimic_wav


class BertModel(torch.nn.Module):
    """Transformer language model.
    """
    def __init__(self, orig_dim=1, embed_dim=256, n_heads=4, max_len=1000, iter=3):
        super().__init__()
        self.iter = iter
        self.embed = DilatedStrideEmbed(orig_dim=orig_dim, embed_dim=embed_dim)

        q_k_func = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim*2, 
                             kernel_size=9, padding= (9-1)//2)

        encoder_layer = TransformerEncoderLayer_CustomAttn(d_model=embed_dim, nhead = n_heads, activation="gelu", 
                                                          custom_attn=None, feedforward="fc", 
                                                          channel_pred=False, ssp=False, 
                                                          q_k_func= q_k_func)
        self.encoder = TransformerEncoder_CustomAttn(encoder_layer, num_layers=2)

        self.mpc = BertMPCHead(orig_dim=orig_dim, embed_dim=embed_dim)

    def forward(self, x, return_attn_weights=False, masktoken_bool=None):
        embedding = self.embed(x) # shape [batch_size, embed_dim, length]
        embedding = embedding.permute(2,0,1) # shape [length, batch_size, embed_dim]
        if return_attn_weights:
            encoded, attn_weights_list = self.encoder(embedding, None, return_attn_weights=return_attn_weights) # shape [length, batch_size, embed_dim]
        else:
            encoded = self.encoder(embedding, None) # shape [length, batch_size, embed_dim]
        encoded = encoded.permute(1,2,0) # shape [batch_size, embed_dim, length]

        mpc_projection = self.mpc(encoded) # shape [batch_size, embed_dim, length]
        mpc_projection = mpc_projection.transpose(1,2) # shape [batch_size, length, embed_dim]

        if return_attn_weights:
            return mpc_projection, attn_weights_list
        else:
            return mpc_projection
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

    def __init__(self, d_model, max_len=512):
        super().__init__()

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
    def __init__(self, orig_dim=12,embed_dim=32):
        super().__init__()
        # output_size=(w+2*pad-(d(k-1)+1))/s+1
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=orig_dim, out_channels=embed_dim, kernel_size=11, stride=1, padding=5, dilation=1),
            nn.BatchNorm1d(num_features=embed_dim, track_running_stats=False),
            nn.ReLU()
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


from torch.optim.lr_scheduler import LambdaLR
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        current_step += 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    