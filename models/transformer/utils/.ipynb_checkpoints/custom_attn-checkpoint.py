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

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, pos_embed: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

    
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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", custom_attn="cosine_similarity"):
        super().__init__()
        self.self_attn = MultiheadAttention_CustomAttn(d_model, nhead, dropout=dropout, custom_attn=custom_attn)
        # Implementation of Feedforward model
        self.linear1 = nn.modules.linear.Linear(d_model, dim_feedforward)
        self.dropout = nn.modules.dropout.Dropout(dropout)
        self.linear2 = nn.modules.linear.Linear(dim_feedforward, d_model)

        self.norm1 = nn.modules.normalization.LayerNorm(d_model)
        self.norm2 = nn.modules.normalization.LayerNorm(d_model)
        self.dropout1 = nn.modules.dropout.Dropout(dropout)
        self.dropout2 = nn.modules.dropout.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_CustomAttn, self).__setstate__(state)

    def forward(self, src: Tensor, pos_embed: Tensor) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, pos_embed)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomAttn_Kernel(nn.Module):
    def __init__(self, custom_attn="cosine_similarity"):
        super().__init__()
        self.custom_attn = custom_attn
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
        if self.custom_attn == "cosine_similarity":
            # modify to custom_attn module in the future, this is legacy
            pos_embed = F.normalize(pos_embed, p=1, dim=1) # shape [batch_size, num_neighbors, length]
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

    def __init__(self, embed_dim, num_heads, dropout=0., custom_attn="cosine_similarity", bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super().__init__()
        #new stuff
        self.custom_attn = custom_attn
        self.custom_attn_kernel = CustomAttn_Kernel(custom_attn = custom_attn)
        #end new stuff
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
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

    def forward(self, query, key, value, pos_embed):
        # replaced Functional's multi_head_pos_sim_attention_forward and baked it in

        tgt_len, bsz, embed_dim = query.size()

        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        # self-attention
        q, k, v = linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q * scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # new positional embedding addition
        if self.custom_attn is not None:
            pos_embedding_sim = self.custom_attn_kernel(pos_embed)
            pos_embedding_sim_softmax = softmax(pos_embedding_sim, dim = -1)
            pos_embedding_sim_softmax = pos_embedding_sim_softmax.repeat(self.num_heads, 1, 1)
            attn_output_weights = torch.mul(attn_output_weights, pos_embedding_sim_softmax)
        #end new stuff

        attn_output_weights = softmax(
            attn_output_weights, dim=-1)

        attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)

        return attn_output, attn_output_weights.sum(dim=1) / self.num_heads

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