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

