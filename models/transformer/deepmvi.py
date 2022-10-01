import torch
import torch.nn as nn
from .utils.custom_convattn_deepmvi import TransformerEncoderLayer_CustomAttn,TransformerEncoder_CustomAttn

class MainModel(torch.nn.Module):
    """Transformer language model.
    """
    def __init__(self, orig_dim=1, embed_dim=256, n_heads=4, max_len=1000, iter=3,kernel_size=21):
        super().__init__()
        self.iter = iter
        self.embed = ConvEmbedding(orig_dim=orig_dim, embed_dim=embed_dim,kernel_size=kernel_size)
        
        #self.deconv = nn.ConvTranspose1d(embed_dim,orig_dim,kernel_size = kernel_size,stride=kernel_size)
        self.deconv = nn.ConvTranspose1d(embed_dim,orig_dim,kernel_size = kernel_size,stride=kernel_size,output_padding=13)
        
        q_k_func = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim*2, 
                             kernel_size=1, padding= 0)

        encoder_layer = TransformerEncoderLayer_CustomAttn(d_model=embed_dim, nhead = n_heads, activation="gelu", 
                                                          custom_attn=None, feedforward="fc", 
                                                          channel_pred=False, ssp=False, 
                                                          q_k_func= q_k_func,max_len=max_len)
        self.encoder = TransformerEncoder_CustomAttn(encoder_layer, num_layers=2)
        
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

    def forward(self, x, return_attn_weights=False, test=False): #shape of x: [batch_size, length, channels]
        embedding = self.embed(x) # shape [batch_size, embed_dim, length]
        embedding = embedding.permute(2,0,1) # shape [length, batch_size, embed_dim]
        if return_attn_weights:
            encoded, attn_weights_list = self.encoder(embedding, None, return_attn_weights=return_attn_weights) # shape [length, batch_size, embed_dim]
        else:
            encoded = self.encoder(embedding, None) # shape [length, batch_size, embed_dim]
        encoded = encoded.permute(1,2,0) # shape [batch_size, embed_dim, length]


        mpc_projection = self.deconv(encoded).transpose(1,2)
        
        #for deep MVI: add local context computation
        local_feats = self.context_feats(x[:,:,0],test).transpose(1,2) #before transpose shape is [bs,1,length]
        
        #fixing the size that changed after deconv and repeat_interleave
        
        mpc_projection = mpc_projection[:,0:local_feats.shape[1],:]
        feats = torch.cat([mpc_projection,local_feats],dim=2) #shape [bs,length,1+org_dim]
        mean = self.mean_outlier_layer(feats) #[:,:,0]
        if return_attn_weights:
            return mean, attn_weights_list #mpc_projection, attn_weights_list
        else:
            return mean #mpc_projection

class ConvEmbedding(torch.nn.Module):
    def __init__(self, orig_dim=12,embed_dim=32,kernel_size=11):
        super().__init__()
        # output_size=(w+2*pad-(d(k-1)+1))/s+1
        self.embedding = nn.Sequential(nn.Conv1d(in_channels=orig_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=kernel_size, padding=1, dilation=1))
        
    def forward(self, x):
        # x stores integers and has shape [batch_size, length, channels]        
        x = x.permute(0,2,1)
        x1 = self.embedding(x)
        return x1