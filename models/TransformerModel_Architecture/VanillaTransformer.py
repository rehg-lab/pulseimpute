import torch
import math
import torch.nn as nn

class MainModel(torch.nn.Module):
    """Transformer language model.
    """
    def __init__(self, orig_dim=1, embed_dim=256, n_heads=4, max_len=1000):
        super().__init__()
        self.embed = ConvEmbedding(orig_dim=orig_dim, embed_dim=embed_dim)
        self.pos_embed = PositionalEmbedding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead = n_heads, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # masked predictive coding head
        self.mpc = Projection(orig_dim=orig_dim, embed_dim=embed_dim)

    def forward(self, x, return_attn_weights=False):
        embedding = self.embed(x) # shape [batch_size, embed_dim, length]
        pos_embedding = self.pos_embed(x)
        embedding = embedding + pos_embedding
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

