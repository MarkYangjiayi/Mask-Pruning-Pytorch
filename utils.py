import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class Mask(nn.Module):
    def __init__(self,bits=1):
        super(Mask, self).__init__()
        self.weight = None
        self.bits = bits

    def quant(self):
        if self.bits == None:
            return self.weight
        else:
           max_quan = 2**(self.bits-1)
           quan_weight = torch.round(max_quan*self.weight)/max_quan
        return quan_weight


    def forward(self,input):
        if type(self.weight) == type(None): #https://github.com/pytorch/pytorch/issues/5486
            self.weight = Parameter(torch.nn.init.constant_(torch.Tensor(input.shape[1:]), val=1.0).cuda())
        result = input*self.quant()
        return result
