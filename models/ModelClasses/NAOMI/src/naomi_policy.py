import torch
import numpy as np
import os
from .model import NAOMI, num_trainable_params
# pretrain policy
# pretrain discriminator
from torch.autograd import Variable
from .helpers import draw_and_stats

# GAN Train


class naomi_policy(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.real_naomi_policy = NAOMI(params)


    def forward(self, data, ground_truth, forward=False, sample=False):
        assert not (forward and sample)
        assert not (not forward and not sample)
        # ground truth  and data starts out as bs, seq_len, dim due to collatefn
        ground_truth = ground_truth.transpose(0,1)
        data = data.transpose(0,1) # seq_len, batch_size, dim
        if forward:
            return self.real_naomi_policy.forward(data, ground_truth)
        if sample:
            data = list(torch.tensor_split(data, data.shape[0]))
            # data_list = []
            # for i in range(data.shape[0]):
            #     data_list.append(data[i:i+1])
            # import pdb; pdb.set_trace()
            samples = self.real_naomi_policy.sample(data)

            # states = samples[:-1, :, :]
            # actions = samples[1:, :, :]
            # exp_states = ground_truth[:-1, :, :]
            # exp_actions = ground_truth[1:, :, :]

            # mod_stats = draw_and_stats(samples.data, name=None, i_iter=None, task=None, draw=False, compute_stats=True, missing_list=None)
            # exp_stats = draw_and_stats(ground_truth.data, name=None, i_iter=None, task=None, draw=False, compute_stats=True, missing_list=None)
    
            # return exp_states.data, exp_actions.data, ground_truth.data, \
            #     states, actions, samples.data, None, None
            return None, None, None, \
                None, None, samples.data, None, None

    