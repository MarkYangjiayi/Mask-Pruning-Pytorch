import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class Mask(nn.Module):
    def __init__(self,bits=1):
        super(Mask, self).__init__()
        #normal random initialization
        #self.weight = torch.nn.init.normal_(torch.Tensor(self.weight_shape),mean=0.5,std=0.1)

        #uniform initialization
        #self.weight = torch.nn.init.uniform(torch.Tensor(self.weight_shape), a=0.0, b=1.0)

        #constant initialization all 1
        #self.weight = torch.nn.init.constant(torch.Tensor(self.weight_shape), val=1.0)
        self.weight = None
        self.bits = bits


    def quant(self):
        if self.bits == None:
            return self.weight
        else:
           max_quan = 2**(self.bits-1)
           quan_weight = torch.round(max_quan*self.weight)/max_quan
        return quan_weight

    #apply mask on every feature in the batch
    def apply_mask(self,input,mask):
        result = input*mask
        return result

    def forward(self,input):
        if type(self.weight) == type(None): #https://github.com/pytorch/pytorch/issues/5486
            # self.weight = Parameter(torch.Tensor(input.shape))
            # self.weight.data.normal_(1.0,0)
            self.weight = torch.nn.init.constant_(torch.Tensor(input.shape[1:]), val=1.0).cuda()
        result = self.apply_mask(input,self.weight)
        return result
