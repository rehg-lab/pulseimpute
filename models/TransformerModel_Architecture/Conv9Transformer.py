import torch
import torch.nn as nn
from .utils.custom_convattn import TransformerEncoderLayer_CustomAttn, TransformerEncoder_CustomAttn

class MainModel(torch.nn.Module):
    """Transformer language model.
    """
    def __init__(self, orig_dim=1, embed_dim=256, n_heads=4, max_len=1000, iter=3):
        super().__init__()
        self.iter = iter
        self.embed = ConvEmbedding(orig_dim=orig_dim, embed_dim=embed_dim)

        q_k_func = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim*2, 
                             kernel_size=9, padding= (9-1)//2)

        encoder_layer = TransformerEncoderLayer_CustomAttn(d_model=embed_dim, nhead = n_heads, activation="gelu", 
                                                          custom_attn=None, feedforward="fc", 
                                                          channel_pred=False, ssp=False, 
                                                          q_k_func= q_k_func)
        self.encoder = TransformerEncoder_CustomAttn(encoder_layer, num_layers=2)

        # masked predictive coding head
        self.mpc = Projection(orig_dim=orig_dim, embed_dim=embed_dim)

    def forward(self, x, return_attn_weights=False):
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

class ConvEmbedding(torch.nn.Module):
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

class Projection(torch.nn.Module):

    def __init__(self, orig_dim=12, embed_dim = 32):
        super().__init__()
        self.projection = nn.Sequential(nn.Conv1d(in_channels=embed_dim, out_channels=orig_dim, kernel_size=11, stride=1, padding=5*1, dilation=1))

    def forward(self, encoded_states):
        original = self.projection(encoded_states)
        return original
