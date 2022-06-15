r"""Functional interface"""
import copy
import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor
# from torch.nn.modules.linear import _LinearWithBias
from torch.nn.modules.linear import Linear as _LinearWithBias

from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch import _VF

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import numpy as np
import math

# from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from longformer.longformer import LongformerSelfAttention
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from longformer.sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])



class Custom_LongformerSelfAttn(LongformerSelfAttention):
    def __init__(self, config, layer_id, q_func=None, k_func=None, q_k_func = None, time_decay=False):
        super().__init__(config, layer_id)
        if q_k_func:
            self.q_k_func = q_k_func
        else:
            if q_func and k_func:
                self.query = q_func
                self.key = k_func
            self.q_k_func = None

        self.query_global = None
        self.key_global = None
        self.value_global = None


    def forward(
        self,
        hidden_states,
        pos_embed=None,
        output_attentions=False,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        '''
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        '''
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        if self.q_k_func:
            q, k= self.q_k_func(hidden_states.permute(1,2,0)).chunk(2, dim=1) # torch.Size([30000, 512, 8])
            q = q.permute(2,0,1)
            k = k.permute(2,0,1)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
        v = self.value(hidden_states)
        q =  q/math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1) # batch_size, seq_len, num_attention_heads, hidden_size = q.size()
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise False
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False)
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        if self.attention_mode == 'tvm':
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_weights are padded with -10000.0 attention scores
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if output_attentions else context_layer
        return outputs

# class TransformerEncoderLayer_CustomAttn(LongformerSelfAttention):
#     def __init__(self, config, layer_id, q_func=None, k_func=None, clone=False, time_decay=False):
#         super().__init__(config, layer_id)
#         if clone:
#             self.query = copy.deepcopy(q_func) 
#             self.key = copy.deepcopy(k_func) 
#         else:
#             if q_func is not None and k_func is not None:
#                 self.query = q_func
#                 self.key = k_func

#         self.query_global = None
#         self.key_global = None
#         self.value_global = None

#         if time_decay:
#             self.log_time_decay_factor = nn.Parameter(torch.rand(1))
#         else:
#             self.log_time_decay_factor = None
    
#     # def forward(
#     #     self,
#     #     hidden_states,
#     #     attention_mask=None,
#     #     output_attentions=False):
        
#     #     return super().forward(hidden_states, attention_mask=attention_mask,  
#     #     is_index_masked = attention_mask < 0, 
#     #     output_attentions=output_attentions)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         output_attentions=False, # put this in front
#         layer_head_mask=None,
#         is_index_masked=None,
#         is_index_global_attn=None,
#         is_global_attn=None,
#     ):
#         """
#         [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
#         *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

#         The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

#             - -10000: no attention
#             - 0: local attention
#             - +10000: global attention
#         """
#         is_index_masked = attention_mask < 0 # new

#         hidden_states = hidden_states.transpose(0, 1)

#         # project hidden states
#         query_vectors = self.query(hidden_states)
#         key_vectors = self.key(hidden_states)
#         value_vectors = self.value(hidden_states)

#         seq_len, batch_size, embed_dim = hidden_states.size()
#         assert (
#             embed_dim == self.embed_dim
#         ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

#         # normalize query
#         query_vectors /= math.sqrt(self.head_dim)

#         query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
#         key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

#         attn_scores = self._sliding_chunks_query_key_matmul(
#             query_vectors, key_vectors, self.one_sided_attn_window_size
#         )
#         if self.log_time_decay_factor:
#             # times = np.arange(self.one_sided_attn_window_size*2+1)
#             # dists = torch.tensor(-squareform(pdist(np.expand_dims(times, axis=1), 'euclidean'))
#             #                     )# [input_len, attend_len]
#             # attn_scores +=  self.log_time_decay_factor.exp()* dists[:-1,:].unsqueeze(1).unsqueeze(0).repeat(batch_size, 1 , self.num_heads,1).float().cuda()
#             times = -torch.abs(torch.arange(-self.one_sided_attn_window_size, self.one_sided_attn_window_size+1))
#             times = times.repeat(batch_size, seq_len, self.num_heads, 1).cuda()
#             attn_scores +=  self.log_time_decay_factor.exp()* times


#         # values to pad for attention probs
#         remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

#         # cast to fp32/fp16 then replace 1's with -inf
#         float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
#             remove_from_windowed_attention_mask, -10000.0
#         )
#         # diagonal mask with zeros everywhere and -inf inplace of padding
#         diagonal_mask = self._sliding_chunks_query_key_matmul(
#             float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
#         )            

#         # pad local attention probs
#         attn_scores += diagonal_mask

#         assert list(attn_scores.size()) == [
#             batch_size,
#             seq_len,
#             self.num_heads,
#             self.one_sided_attn_window_size * 2 + 1,
#         ], f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

#         # compute local attention probs from global attention keys and contact over window dim
#         if is_global_attn:
#             # compute global attn indices required through out forward fn
#             (
#                 max_num_global_attn_indices,
#                 is_index_global_attn_nonzero,
#                 is_local_index_global_attn_nonzero,
#                 is_local_index_no_global_attn_nonzero,
#             ) = self._get_global_attn_indices(is_index_global_attn)
#             # calculate global attn probs from global key

#             global_key_attn_scores = self._concat_with_global_key_attn_probs(
#                 query_vectors=query_vectors,
#                 key_vectors=key_vectors,
#                 max_num_global_attn_indices=max_num_global_attn_indices,
#                 is_index_global_attn_nonzero=is_index_global_attn_nonzero,
#                 is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
#                 is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
#             )
#             # concat to local_attn_probs
#             # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
#             attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

#             # free memory
#             del global_key_attn_scores

#         attn_probs = nn.functional.softmax(
#             attn_scores, dim=-1, dtype=torch.float32
#         )  # use fp32 for numerical stability
#         if layer_head_mask is not None:
#             assert layer_head_mask.size() == (
#                 self.num_heads,
#             ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
#             attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

#         # softmax sometimes inserts NaN if all positions are masked, replace them with 0
#         attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
#         attn_probs = attn_probs.type_as(attn_scores)

#         # free memory
#         del attn_scores

#         # apply dropout
#         attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

#         value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

#         # compute local attention output with global attention value and add
#         if is_global_attn:
#             # compute sum of global and local attn
#             attn_output = self._compute_attn_output_with_global_indices(
#                 value_vectors=value_vectors,
#                 attn_probs=attn_probs,
#                 max_num_global_attn_indices=max_num_global_attn_indices,
#                 is_index_global_attn_nonzero=is_index_global_attn_nonzero,
#                 is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
#             )
#         else:
#             # compute local attn only
#             attn_output = self._sliding_chunks_matmul_attn_probs_value(
#                 attn_probs, value_vectors, self.one_sided_attn_window_size
#             )

#         assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
#         attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

#         # compute value for global attention and overwrite to attention output
#         # TODO: remove the redundant computation
#         if is_global_attn:
#             global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
#                 hidden_states=hidden_states,
#                 max_num_global_attn_indices=max_num_global_attn_indices,
#                 layer_head_mask=layer_head_mask,
#                 is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
#                 is_index_global_attn_nonzero=is_index_global_attn_nonzero,
#                 is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
#                 is_index_masked=is_index_masked,
#             )

#             # get only non zero global attn output
#             nonzero_global_attn_output = global_attn_output[
#                 is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
#             ]

#             # overwrite values with global attention
#             attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
#                 len(is_local_index_global_attn_nonzero[0]), -1
#             )
#             # The attention weights for tokens with global attention are
#             # just filler values, they were never used to compute the output.
#             # Fill with 0 now, the correct values are in 'global_attn_probs'.
#             attn_probs[is_index_global_attn_nonzero] = 0

#         outputs = (attn_output.transpose(0, 1),)

#         if output_attentions:
#             outputs += (attn_probs,)

#         return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs




# class TransformerEncoder_CustomAttn(nn.Module):
#     r"""TransformerEncoder is a stack of N encoder layers

#     Args:
#         encoder_layer: an instance of the TransformerEncoderLayer() class (required).
#         num_layers: the number of sub-encoder-layers in the encoder (required).
#         norm: the layer normalization component (optional).

#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = transformer_encoder(src)
#     """
#     __constants__ = ['norm']

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def set_endofseqidx(self, endofseqidx):
#         for layer in self.layers:
#             layer.set_endofseqidx(endofseqidx)

#     def forward(self, src: Tensor, attention_mask: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, 
#         return_attn_weights=False) -> Tensor:
#         r"""Pass the input through the encoder layers in turn.

#         Args:
#             src: the sequence to the encoder (required).
#             mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """

#         attn_weights_list = []
#         for mod in self.layers:
#             out = mod(src, attention_mask, return_attn_weights)
#             if return_attn_weights:
#                 src = out[0]
#                 attn_weights_list.append(out[1])
#             else:
#                 src = out[0]
#         if self.norm is not None:
#             src = self.norm(src)

#         if return_attn_weights:
#             return src, attn_weights_list
#         else:
#             return src
