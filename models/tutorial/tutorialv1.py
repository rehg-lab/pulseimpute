
import torch
import torch.nn as nn


class MainModel(torch.nn.Module): 
    """Simple LSTM model
    """
    def __init__(self, orig_dim=1, embed_dim=16, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(orig_dim,embed_dim,n_layers,batch_first=True) 
        self.fc = nn.Linear(embed_dim,orig_dim)


    def forward(self, x): #shape: [batch_size, length, orig_dim]
        lstm_out,_ = self.lstm(x)
        fc_out = self.fc(lstm_out)

        return fc_out

