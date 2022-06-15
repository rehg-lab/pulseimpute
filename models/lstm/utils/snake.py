import torch.nn as nn
import random
import torch

def snake_weight_init(m):
#     if hasattr(m, "bias"):
#         m.bias.data.fill_(0)
    if hasattr(m, "weight"):
        if type(m) == nn.LayerNorm:
            return
        torch.div(torch.nn.init.kaiming_uniform_(m.weight), (0.5)**(0.5)) # special initialization for snake
        
class Snake(nn.Module):
    '''
    Applies the Snake Linear Unit  function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/2006.08195.pdf
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.alpha = nn.parameter.Parameter(torch.tensor(10.0, dtype = torch.float32)) 
        self.alpha.data.fill_(.08373) # 80 bpm --> 75 centiseconds per beat --> 2 * pi / 75
        self.alpha.requiresGrad = False

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input - 1/(2 * self.alpha) * (torch.cos(2 * self.alpha * input)) + 1/(2 * self.alpha)