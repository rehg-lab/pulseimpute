r"""Functional interface"""
import copy
import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch import _VF

import gpytorch
import numpy as np

class NearestNeighborsEmbed(torch.nn.Module):

    def __init__(self, in_channels=1, window_size=49):
        super().__init__()
        self.avg_layer = nn.Conv1d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_layer.weight.data.fill_(1/in_channels)
        self.avg_layer.weight.requires_grad = False
        
        self.nearby_neighbors_layer = nn.Conv1d(1, window_size, kernel_size=window_size, stride=1, padding=window_size//2, bias=False)
        self.nearby_neighbors_layer.weight = nn.Parameter(torch.eye(window_size).unsqueeze(1))
        self.nearby_neighbors_layer.weight.requires_grad = False
        
    def forward(self, src):
        src = src.transpose(2,1)
        avg_src = self.avg_layer(src)
        nn_src = self.nearby_neighbors_layer(avg_src)
        nn_src = nn_src
        return nn_src
    
class TransformerEncoder_CustomAttn(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, clstoken=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.clstoken = clstoken

    def set_endofseqidx(self, endofseqidx):
        for layer in self.layers:
            layer.set_endofseqidx(endofseqidx)

    def forward(self, src: Tensor, pos_embed: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, 
        return_attn_weights=False) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if self.clstoken:
            attn_weights_list = []
            for mod in self.layers:
                out = mod(src, pos_embed, return_attn_weights) #error will be thrown if return_attn_weights is on...
                if return_attn_weights:
                    src = out[0]
                    cls_vector = out[1]
                    attn_weights_list.append(out[2])
                else:
                    src = out[0]
                    cls_vector = out[1]
            if self.norm is not None:
                src = self.norm(src)
                cls_vector = self.norm(cls_vector)
            if return_attn_weights:
                return src, cls_vector, attn_weights_list
            else:
                return src, cls_vector
        else:
            attn_weights_list = []
            for mod in self.layers:
                out = mod(src, pos_embed, return_attn_weights)
                if return_attn_weights:
                    src = out[0]
                    attn_weights_list.append(out[1])
                else:
                    src = out
            if self.norm is not None:
                src = self.norm(src)

            if return_attn_weights:
                return src, attn_weights_list
            else:
                return src

    
class TransformerEncoderLayer_CustomAttn(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", feedforward="fc",
                 custom_attn="cosine_similarity", channel_pred=False, ssp=False, bottleneck_size=None,nb_filters=32,pos_embed_outside=False, 
                 qk_kernel_all="linear", qk_kernel_cls="linear", rpe_dim=0, clstoken=True,incep_depth=6,incep_bugfix=False, 
                 factkern_bugfix=False, factkern_bugfix_2=False, bn_running=True, sspmask=True, maskfix=False,
                pos_learnable_dim=None, pos_scaling=False):
        super().__init__()
        # new stuff here
        self.ssp = ssp
        self.self_attn = MultiheadAttention_CustomAttn(d_model, nhead, dropout=dropout, custom_attn=custom_attn, 
                                                        channel_pred=channel_pred, ssp=ssp, pos_embed_outside=pos_embed_outside,
                                                        qk_kernel=qk_kernel_all, rpe_dim=rpe_dim, clstoken=clstoken, 
                                                        factkern_bugfix=factkern_bugfix, factkern_bugfix_2=factkern_bugfix_2,
                                                        sspmask=sspmask, maskfix=maskfix, pos_learnable_dim=pos_learnable_dim, pos_scaling=pos_scaling)
        self.channel_pred = channel_pred
        self.clstoken = clstoken
        # Implementation of Feedforward model
        self.feedforward = feedforward
        if self.feedforward =="fc":
            self.linear1 = nn.modules.linear.Linear(d_model, dim_feedforward)
            self.dropout = nn.modules.dropout.Dropout(dropout)
            self.linear2 = nn.modules.linear.Linear(dim_feedforward, d_model)
        elif self.feedforward == "inception":
            if bottleneck_size == None:
                bottleneck_size = d_model
            self.inception = Inception1d(input_channels=d_model,use_residual=True, kernel_size=8*5, depth=incep_depth, 
                bottleneck_size=bottleneck_size, nb_filters=nb_filters,incep_bugfix=incep_bugfix, bn_running=bn_running)
            if self.clstoken:
                self.cls_self_attn = MultiheadAttention_CustomAttn(d_model, nhead, dropout=dropout, custom_attn=custom_attn, 
                                                               pos_embed_outside=pos_embed_outside,
                                                               query_onlycls=True, qk_kernel=qk_kernel_cls, rpe_dim=rpe_dim)          

        self.norm1 = nn.modules.normalization.LayerNorm(d_model)
        self.norm2 = nn.modules.normalization.LayerNorm(d_model)
        self.dropout1 = nn.modules.dropout.Dropout(dropout)
        self.dropout2 = nn.modules.dropout.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_CustomAttn, self).__setstate__(state)

    def set_endofseqidx(self, endofseqidx):
        self.self_attn.set_endofseqidx(endofseqidx)

    def forward(self, src: Tensor, pos_embed: Tensor, return_attn_weights=False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required). shape [length, batch_size, embed_dim]
            pos_embed: the positional embedding to the encoder layer (required). shape [batch_size, embed_dim, length]
        Shape:
            see the docs in Transformer class.
        """
        if (self.ssp or self.channel_pred) and self.clstoken:
            cls_vector = torch.zeros(1,src.shape[1],src.shape[2]).cuda()
            src = torch.cat((cls_vector, src), 0)
        src2, attn_weights = self.self_attn(src, src, src, pos_embed)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        if self.feedforward == "fc":
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        elif self.feedforward == "inception":
            if self.clstoken:
                # changed all src to src2
                cls_vector = src[0, :, :].unsqueeze(0)
                src = src[1:, :, :] # shape [len, batch_size, channels]
                src2 = self.inception(src.permute(1,2,0)) # shape [batch_size, channels, len]
                src2 = src2.permute(2, 0, 1)
                src2 = torch.cat((cls_vector, src2), 0)
                cls_vector = self.cls_self_attn(src2, src2, src2, pos_embed=torch.ones(src2.shape[1], 1, src2.shape[0]).cuda())[0]
                src2 = torch.cat((cls_vector, src2[1:, :, :]), 0)
            else:
                src2 = self.inception(src.permute(1,2,0)) # shape [batch_size, channels, len]
                src2 = src2.permute(2, 0, 1)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if (self.ssp or self.channel_pred) and self.clstoken:
            cls_vector = src[0, :, :]
            src = src[1:, :, :]
            return src, cls_vector
        else:
            if return_attn_weights:
                return src, attn_weights
            else:
                return src


class MultiheadAttention_CustomAttn(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., custom_attn="cosine_similarity", 
                bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                channel_pred=False, ssp=False, query_onlycls=False, pos_embed_outside=False,
                qk_kernel = "linear", rpe_dim=0,clstoken=True,factkern_bugfix=False, factkern_bugfix_2=False, sspmask=True, maskfix=False,
                pos_learnable_dim=None, pos_scaling=False):
        super().__init__()
        #new stuff
        self.ssp = ssp
        if not self.ssp:
            sspmask=False
        self.channel_pred = channel_pred
        self.query_onlycls = query_onlycls
        self.custom_attn = custom_attn
        if pos_learnable_dim:
            self.custom_attn_kernel = CustomAttn_Kernel(custom_attn = custom_attn, pos_learnable_dim=pos_learnable_dim) # pos_embed.shape[1] gets us window size i think
        else:
            self.custom_attn_kernel = CustomAttn_Kernel(custom_attn = custom_attn)
        self.pos_embed_outside=pos_embed_outside
        self.endofseqidx = None
        self.qk_kernel = qk_kernel
        self.rpe_dim = rpe_dim
        self.clstoken = clstoken
        self.factkern_bugfix = factkern_bugfix
        self.factkern_bugfix_2=factkern_bugfix_2
        self.sspmask = sspmask
        self.maskfix = maskfix
        self.pos_scaling = pos_scaling
        # if self.qk_kernel == "rbf":
        #     self.qk_kernel_module = gpytorch.kernels.RBFKernel()
        #end new stuff
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim 
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self.rpe_dim > 0:
            self.rpe_embeddings_key = Parameter(torch.Tensor(rpe_dim*2+1, self.head_dim ))
            xavier_uniform_(self.rpe_embeddings_key)
            self.rpe_embeddings_val = Parameter(torch.Tensor(rpe_dim*2+1, self.head_dim ))
            xavier_uniform_(self.rpe_embeddings_val)
            
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else: # with default settings, it is always going here
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention_CustomAttn, self).__setstate__(state)

    def set_endofseqidx(self, endofseqidx):
        self.endofseqidx = endofseqidx

    def forward(self, query, key, value, pos_embed):
        # replaced Functional's multi_head_pos_sim_attention_forward and baked it in

        tgt_len, bsz, embed_dim = query.size()

        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        # self-attention
        q, k, v = linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if self.query_onlycls:
            q = q[:, 0, :].unsqueeze(1) # shape [embed, length, batch_size]
            tgt_len = 1
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.qk_kernel == "linear":
            q = q * scaling
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        elif self.qk_kernel == "rbf":
            attn_output_weights = - scaling *  torch.cdist(q, k, p=2.0) ** 2
        elif self.qk_kernel == "periodic":
            q1 = q.unsqueeze(1)
            k1 = k.unsqueeze(2)
            deltas = torch.abs(q1 - k1)
            l1norm = np.pi *  torch.sum(deltas, dim=-1)/ nn.Parameter(torch.tensor([1.0])).cuda()
            attn_output_weights = scaling * torch.square(torch.sin(l1norm)) / nn.Parameter(torch.tensor([1.0])).cuda()**2

        if not self.query_onlycls and self.rpe_dim > 0:
            relations_keys, relations_values = self._generate_relative_positions_embeddings(
                length=src_len, max_relative_position=self.rpe_dim)
            q_withrpe_k = torch.bmm(q.transpose(0,1), relations_keys.transpose(1,2)) # shape len, bsz*num_heads, len
            q_withrpe_k = q_withrpe_k.transpose(0,1)
            q_withrpe_k[:, 0, :] = 0 # setting all position embeddings piece wise associated with cls are 0
            q_withrpe_k[:, :, 0] = 0
            if self.qk_kernel != "linear":
                q_withrpe_k = q_withrpe_k * scaling

            attn_output_weights = attn_output_weights + q_withrpe_k

        # new positional embedding addition
        if self.custom_attn is not None: #fix this too because self.ssp shouldnt be dependent on custom attention sigh
            if self.query_onlycls:
                pos_embedding_sim = pos_embed
            else:
                pos_embedding_sim = self.custom_attn_kernel(pos_embed) # [batch_size, input_len, attend_len]
                if self.pos_scaling:
                    pos_embedding_sim = pos_embedding_sim * scaling
                # need to modify here ...
                if self.ssp:
                    # mask out second sequence from first sequence and vice versa
                    mask = torch.ones(pos_embedding_sim.shape).cuda()
                    mask[:,0:mask.shape[1]//2,mask.shape[2]//2:mask.shape[2]] = 0
                    mask[:,mask.shape[1]//2:mask.shape[1],0:mask.shape[2]//2] = 0
                    if not self.maskfix and self.sspmask:
                        pos_embedding_sim = torch.mul(pos_embedding_sim, mask)
                if (self.ssp or self.channel_pred) and self.clstoken:
                    cls_pos_embed_sim = torch.ones(pos_embedding_sim.shape[0], pos_embedding_sim.shape[1], 1).cuda()
                    pos_embedding_sim = torch.cat((cls_pos_embed_sim, pos_embedding_sim), 2)

                    cls_pos_embed_sim = torch.ones(pos_embedding_sim.shape[0], 1, pos_embedding_sim.shape[2]).cuda()
                    pos_embedding_sim = torch.cat((cls_pos_embed_sim, pos_embedding_sim), 1) # this should be now [input_len+1, input_len+1] and cls in beginning
            if not self.factkern_bugfix:
                pos_embedding_sim_softmax = softmax(pos_embedding_sim, dim = -1)
                pos_embedding_sim_softmax = pos_embedding_sim_softmax.repeat(self.num_heads, 1, 1) # needed so bmm works
            else:
                pos_embedding_sim = pos_embedding_sim.repeat(self.num_heads, 1, 1)
            if not self.pos_embed_outside:
                attn_output_weights = torch.mul(attn_output_weights, pos_embedding_sim_softmax)
                
        else:
            if (self.ssp or self.channel_pred) and self.clstoken:
                shape = (q.shape[0]//self.num_heads, q.shape[1]-1, q.shape[1]-1)
            else:
                shape = (q.shape[0]//self.num_heads, q.shape[1], q.shape[1])
            pos_embedding_sim_softmax = torch.ones(shape).cuda()
            if self.ssp:
                mask = torch.ones(shape).cuda()
                mask[:,0:mask.shape[1]//2,mask.shape[2]//2:mask.shape[2]] = 0
                mask[:,mask.shape[1]//2:mask.shape[1],0:mask.shape[2]//2] = 0
                if not self.maskfix and self.sspmask:
                    pos_embedding_sim_softmax = torch.mul(pos_embedding_sim_softmax, mask)
            if (self.ssp or self.channel_pred) and self.clstoken:
                cls_pos_embed_sim = torch.ones(shape[0], shape[1], 1).cuda()
                pos_embedding_sim_softmax = torch.cat((cls_pos_embed_sim, pos_embedding_sim_softmax), 2)

                cls_pos_embed_sim = torch.ones(shape[0], 1, shape[2]+1).cuda()
                pos_embedding_sim_softmax = torch.cat((cls_pos_embed_sim, pos_embedding_sim_softmax), 1)
            pos_embedding_sim_softmax = pos_embedding_sim_softmax.repeat(self.num_heads, 1, 1) # needed so bmm works
            if not self.pos_embed_outside:
                attn_output_weights = torch.mul(attn_output_weights, pos_embedding_sim_softmax)

        if self.qk_kernel == "linear":
            if self.factkern_bugfix and self.factkern_bugfix_2:
                temp = attn_output_weights + pos_embedding_sim
                if self.maskfix and self.sspmask:
                    temp = torch.mul(attn_output_weights + pos_embedding_sim, mask.repeat(self.num_heads, 1, 1)) # remove 0s from masks
                logsumexp = torch.logsumexp(temp, dim=-1, keepdim=True)
                attn_output_weights = attn_output_weights + pos_embedding_sim - logsumexp
                attn_output_weights = torch.exp(attn_output_weights)
            else:
                attn_output_weights = softmax(attn_output_weights, dim=-1)
        elif self.qk_kernel == "rbf" or self.qk_kernel == "periodic" :
            # log weights are the log softmax of them
            # attn_output_weights are the terms inside_exp https://blog.feedly.com/tricks-of-the-trade-logsumexp/
            if self.factkern_bugfix:
                if self.factkern_bugfix_2:
                    temp = attn_output_weights + pos_embedding_sim
                    if self.maskfix and self.sspmask:
                        temp = torch.mul(attn_output_weights + pos_embedding_sim, mask.repeat(self.num_heads, 1, 1)) # remove 0s from masks
                    logsumexp = torch.logsumexp(temp, dim=-1, keepdim=True)
                    attn_output_weights = attn_output_weights + pos_embedding_sim - logsumexp
                else: # this is completely wrong, it should be logsum * (periodic kernel) * exp(input)
                    attn_output_weights = attn_output_weights - torch.logsumexp(
                        torch.mul(pos_embedding_sim, attn_output_weights), dim=-1, keepdim=True)
            else:
                attn_output_weights = attn_output_weights - torch.logsumexp(attn_output_weights, dim=-1, keepdim=True)
            attn_output_weights = torch.exp(attn_output_weights)
            if self.factkern_bugfix and not self.factkern_bugfix_2:
                attn_output_weights = torch.mul(attn_output_weights, pos_embedding_sim)


        if self.pos_embed_outside and not self.factkern_bugfix:
            attn_output_weights = torch.mul(attn_output_weights, pos_embedding_sim_softmax)

        if self.maskfix and self.sspmask:
            attn_output_weights = torch.mul(attn_output_weights, mask.repeat(self.num_heads, 1, 1))
        #end new stuff
        attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_output_weights, v)

        if not self.query_onlycls and self.rpe_dim > 0:
            attn_output_weights_withrpe_v = torch.bmm(attn_output_weights.transpose(0,1), relations_values) # shape len, bsz*num_heads, len
            attn_output_weights_withrpe_v = attn_output_weights_withrpe_v.transpose(0,1)
            attn_output = attn_output + attn_output_weights_withrpe_v

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)

        return attn_output, attn_output_weights.sum(dim=1) / self.num_heads

    def _generate_relative_positions_matrix(self, length, max_relative_position):
        """Generates matrix of relative positions between inputs."""
        range_vec = torch.arange(0, length)
        range_mat = torch.reshape(range_vec.repeat(length), (length, length))
        distance_mat = range_mat - torch.transpose(range_mat, 0, 1)
        distance_mat_clipped = torch.clamp(distance_mat, min=-max_relative_position,
                                        max=max_relative_position)
        # Shift values to be >= 0. Each integer still uniquely identifies a relative
        # position difference.
        final_mat = distance_mat_clipped + max_relative_position
        return final_mat

    def _generate_relative_positions_embeddings(self, length, max_relative_position):
        """Generates tensor of size [length, length, depth]."""
        relative_positions_matrix = self._generate_relative_positions_matrix(length, max_relative_position)
        # Generates embedding for each relative position of dimension depth
        relations_keys = self.rpe_embeddings_key[relative_positions_matrix] # hopefully this will work lmao
        relations_values = self.rpe_embeddings_val[relative_positions_matrix]
        return relations_keys, relations_values

class CustomAttn_Kernel(nn.Module):
    def __init__(self, custom_attn="cosine_similarity", pos_learnable_dim=None):
        super().__init__()
        self.custom_attn = custom_attn
        if pos_learnable_dim:
            self.pos_learnable = True
            self.pos_linear = _LinearWithBias(pos_learnable_dim, pos_learnable_dim)

        if self.custom_attn == "rbf":
            self.custom_attn_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif self.custom_attn == "periodic": # 80 beats per minute --> 75 centiseconds per beat
            # according to wikipedia, std_Dev of 10 beats per minute --> 600 centiseconds per beat ((10/60/100)^-1)
            self.custom_attn_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(period_length_prior=
                                                                                                   gpytorch.priors.NormalPrior(loc=75, scale=20)))
#             self.custom_attn_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(period_length_prior=None))
        elif self.custom_attn == "locally_periodic_mul" : 
            self.periodic =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(period_length_prior=
                                                                                                   gpytorch.priors.NormalPrior(loc=75, scale=20)))
            self.rbf = gpytorch.kernels.RBFKernel()
        elif self.custom_attn == "locally_periodic_add":
            self.periodic =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
            self.rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, pos_embed: Tensor) -> Tensor: # pos_embed start in shape [batch_size, num_neighbors, length]
        if self.pos_learnable:
            pos_embed = pos_embed.transpose(1,2)
            pos_embed = self.pos_linear(pos_embed)
            pos_embed = pos_embed.transpose(1,2)

        if self.custom_attn == "cosine_similarity":
            # modify to custom_attn module in the future, this is legacy
            pos_embed = F.normalize(pos_embed, p=1, dim=1) # shape [batch_size, num_neighbors, length]
            pos_embedding_sim = torch.bmm(pos_embed.transpose(1,2), pos_embed) # shape [batch_size, length, length]
        elif self.custom_attn == "linear":
            pos_embedding_sim = torch.bmm(pos_embed.transpose(1,2), pos_embed) # shape [batch_size, length, length]
        elif self.custom_attn == "locally_periodic_mul":
            pos_embed = pos_embed.transpose(1,2)
            periodic = self.periodic(pos_embed)
            rbf = self.rbf(pos_embed)
            pos_embedding_sim = periodic.mul(rbf).evaluate()
        elif self.custom_attn == "locally_periodic_add":
            pos_embed = pos_embed.transpose(1,2)
            periodic = self.periodic(pos_embed)
            rbf = self.rbf(pos_embed)
            pos_embedding_sim = (periodic + rbf).evaluate()
        else:
            lazy_pos_embedding_sim = self.custom_attn_module(pos_embed.transpose(1,2)) # shape [batch_size, length, length]
            pos_embedding_sim = lazy_pos_embedding_sim.evaluate()
        
        return pos_embedding_sim

def noop(x): return x
class InceptionBlock1d(nn.Module):
    def __init__(self, in_channels, nb_filters, kss, stride=1, act='linear', bottleneck_size=32, bn_running=True):
        super().__init__()
        if bottleneck_size > 0:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, stride=stride,
                     padding=(1-1)//2, bias=False)
            self.convs = nn.ModuleList([nn.Conv1d(bottleneck_size, nb_filters, kernel_size=ks, stride=stride, padding=(ks-1)//2, bias=False)
            for ks in kss])
        else:
            self.bottleneck = noop
            self.convs = nn.ModuleList([nn.Conv1d(in_channels, nb_filters, kernel_size=ks, stride=stride, padding=(ks-1)//2, bias=False)
            for ks in kss])
        

        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), 
                                         nn.Conv1d(in_channels, nb_filters, kernel_size=1, stride=stride, padding=(1-1)//2, bias=False))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters, track_running_stats=bn_running), nn.ReLU())

        self.bn = nn.BatchNorm1d((len(kss)+1)*nb_filters, track_running_stats=bn_running)

    def forward(self, x):
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))
        return out

class Shortcut1d(nn.Module):
    def __init__(self, ni, nf, bn_running=True):
        super().__init__()
        self.act_fn=nn.ReLU(True)
        self.conv=nn.Conv1d(ni, nf, kernel_size=1, stride=1, padding=(1-1)//2, bias=False)
        self.bn=nn.BatchNorm1d(nf, track_running_stats=bn_running)

    def forward(self, inp, out): 
        return self.act_fn(out + self.bn(self.conv(inp)))

class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual, incep_bugfix=False,bn_running=True):
        super().__init__()
        self.depth = depth
        # assert((depth % 3) == 0) commenting this out lmaooo hope this doesnt break anything
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList()
        self.sk = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                in_dim = input_channels
            else:
                in_dim = n_ks*nb_filters
            self.im.append(InceptionBlock1d(in_dim, nb_filters=nb_filters, kss=kss, bottleneck_size=bottleneck_size,bn_running=bn_running))
            if incep_bugfix:
                if (d + 1) // 3 == 0: # oops forgot this bad boy
                    self.sk.append(Shortcut1d(in_dim, n_ks*nb_filters,bn_running=bn_running))
            else: # legacy code for supporting archived code sorry
                self.sk.append(Shortcut1d(in_dim, n_ks*nb_filters, bn_running=bn_running))
        
    def forward(self, x):
        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d//3])(input_res, x)
                input_res = x.clone()
        return x

class Inception1d(nn.Module):
    '''inception time architecture'''
    def __init__(self, input_channels=8, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True,incep_bugfix=False,bn_running=True):
        super().__init__()
        assert(kernel_size>=40)
        kernel_size = [k-1 if k%2==0 else k for k in [kernel_size,kernel_size//2,kernel_size//4]] #was 39,19,9
        
        self.inception = InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth, 
        bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual,incep_bugfix=incep_bugfix,bn_running=bn_running)

    def forward(self,x):
        return self.inception(x)
    
    def get_layer_groups(self):
        depth = self.layers[0].depth
        if(depth>3):
            return ((self.layers[0].im[3:],self.layers[0].sk[1:]),self.layers[-1])
        else:
            return (self.layers[-1])

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def linear(input, weight, bias=None):
    # tens_ops = (input, weight)
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def dropout(input, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.
    See :class:`~torch.nn.Dropout` for details.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    # if not torch.jit.is_scripting():
    #     if type(input) is not Tensor and has_torch_function((input,)):
    #         return handle_torch_function(
    #             dropout, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.dropout_(input, p, training)
            if inplace
            else _VF.dropout(input, p, training))