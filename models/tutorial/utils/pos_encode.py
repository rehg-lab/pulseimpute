import torch
import torch.nn as nn

class sin_positional_encoding(nn.Module):
    # input will be size [batch_size, length, dimensionality]
    def __init__(self, input_len=1000, freq_dims=32, learnable = False):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.positions = torch.arange(0, input_len, dtype = torch.float32).unsqueeze(1).cuda()

        self.freq = torch.FloatTensor(1, freq_dims)
        for i in range(32):
            self.freq[0,i] = (i+1)*.005
        self.freq = nn.parameter.Parameter(self.freq).cuda()
        if learnable:
            self.freq.requiresGrad = True
        else:
            self.freq.requiresGrad = True
        
        # self.phaseshift = nn.parameter.Parameter(torch.FloatTensor(freq_dims).uniform_(0, 100)).cuda()
        # self.phaseshift.requiresGrad = True

    def forward(self):
        '''
        Forward pass of the function.
        '''
        pos_embedding = torch.sin(torch.matmul(self.positions, self.freq)).T
        return pos_embedding.unsqueeze(0)