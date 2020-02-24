import torch
import torch.nn as nn
from torch.nn import functional as F#for Conv2d and ReLU
from torch.optim.optimizer import Optimizer, required#for SGD
from torch.autograd import Function
from torch.nn.parameter import Parameter

import math
import numpy as np
from quantization import *

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

        #stride
        #s1 for line; s2 for column
        if isinstance(stride,int) or len(stride) == 1:
            self.s1 = stride
            self.s2 = stride
        elif len(stride) == 2:
            self.s1 = stride[0]
            self.s2 = stride[1]
        #padding
        self.p = padding

        #kernel size
        if isinstance(kernel_size,int) or len(kernel_size) == 1:
            self.k1 = kernel_size
            self.k2 = kernel_size
        elif len(kernel_size) == 2:
            self.k1 = kernel_size[0]
            self.k2 = kernel_size[1]

        #output feature size
        self.output_H = None
        self.output_W = None

        self.input_shape = None


    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        self.input_shape = input.shape
        return self.conv2d_forward(input,quantizeW(self.weight))#Jiayi: temporarily removed quantization for testing reasons


    def get_feat_ID(self,weight_position):

        '''
        :param input: conv input
        :param weight_position: the weight relative position in the kernel, the first point in the top left is (0,0)
        :return: the corresponding feature position index of weight in the position of given weight_position
        '''


        #input_shape = input.shape
        input_shape = self.input_shape
        assert len(input_shape) == 4
        assert len(weight_position) == 2

        input_C = input_shape[1]
        input_H = input_shape[2]
        input_W = input_shape[3]

        self.output_H = np.int(np.floor((input_H + 2*self.p-self.k1)/self.s1) + 1)
        self.output_W = np.int(np.floor((input_W + 2*self.p-self.k2)/self.s2) + 1)

        #first weight (0,0)
        H_ID_base = []
        W_ID_base = []

        for i in range(self.output_H-weight_position[0]):
            H_ID_base.append(self.s1*i)

        for i in range(self.output_W-weight_position[1]):
            W_ID_base.append(self.s2*i)

        H_ID = [h + weight_position[0] for h in H_ID_base]
        W_ID = [w + weight_position[1] for w in W_ID_base]
        # print(weight_position)
        # print((input_H - self.s1*self.output_H + weight_position[0])//self.s1)
        # TODO:add code here to exclude where exceeds the mask range
        # (input_H - s1*output_H)/s1 (maybe needs some tuning in boundary conditions)

        HW_position = np.zeros([len(H_ID)*len(W_ID),2]).astype(np.int)

        for i, h in enumerate(H_ID):
            for j, w in enumerate(W_ID):
                HW_position[i*len(W_ID)+j][0] = np.int(h)
                HW_position[i*len(W_ID)+j][1] = np.int(w)
        return HW_position

class ReLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return quantizeAE(F.relu(input, inplace=self.inplace))

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

class Mask(nn.Module):
    def __init__(self,bits=1):
        super(Mask, self).__init__()
        self.mask = None
        self.bits = bits

        self.input = None

        self.weight_shape = None
        self.channel = None
        self.height = None
        self.width = None

        self.gl_list = []

    def quant(self):
        if self.bits == None:
            return self.mask
        else:
           max_quan = 2**(self.bits-1)
           quan_weight = torch.round(max_quan*self.mask)/max_quan
        return quan_weight

    def forward(self,input):
        if type(self.mask) == type(None):
        # if self.mask.size() == torch.Size([2]):
            self.mask = Parameter(torch.Tensor(input.shape[1:]).cuda(None))
            nn.init.constant_(self.mask,val=1.0)
            self.weight_shape = input.shape[1:]
            self.channel = input.shape[1]
            self.height = input.shape[2]
            print("height:" + str(self.height))
            self.width = input.shape[3]
        self.input = input.detach()#to keep track of the value in gl regularization
        result = input*self.mask
        return result

class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                #
                # d_p.mul_(group['lr'])
                # d_p = d_p
                # p.data.add_(-1, d_p)
                d_p = qu(qg(d_p)*(-group['lr']))
                p.data.add_(d_p)

        return loss

#get ready for regularization loss
#divide the feat_ID into groups according to structure paramteter
def array_split(feat_ID, stru_param):

    array_shape = feat_ID.shape
    length = array_shape[0]
    import numpy as np
    group_num = np.int(np.floor(length/np.float(stru_param)))
    all_group_list = []
    for i in range(group_num):
        small_group_list = []
        for j in range(stru_param):
            small_group_list.append(feat_ID[i*stru_param+j])
        all_group_list.append(small_group_list)

    return all_group_list

def gl_loss(model, epoch, rate = 0.002):#0.00001
    """
    Function for calculating total gl_loss.
    """
    if rate == 0.: return 0
    rate = adjust_gl_rate(rate, epoch)

    gl_list = []
    conv_layer = None
    mask_layer = None

    # for name, layer in model.named_modules():
    #     if "mask" in name:
    #         mask_layer = layer
    #     if "conv" in name:
    #         conv_layer = layer
    #         gl_layer_test(mask_layer,conv_layer,gl_list)

    for name, parameter in model.named_parameters():
        if "mask" in name:
            print(parameter)

    # for sequence in model.children():
    #     for sub_sequence in sequence.children():
    #         if isinstance(sub_sequence, nn.Sequential):
    #             for layer in sub_sequence.children():
    #                 if isinstance(layer, Mask):
    #                     mask_layer = layer
    #                 if isinstance(layer, Conv2d):
    #                     conv_layer = layer
    #                     # gl_list.append(gl_layer_test(mask_layer,conv_layer))
    #                     gl_layer_test(mask_layer,conv_layer,gl_list)
    # return torch.sum(torch.stack(gl_list)).mul_(rate)
    return torch.sum(torch.stack(gl_list)).mul_(rate)

def gl_layer_test(mask_layer, conv_layer, gl_list, stru_param=4):
    """
    Function for calculating single layer loss.
    '''
    :param mask_layer: mask layer which contains the structure weights
    :param conv_layer: the consecutive conv layer
    :param conv_layer: the number of elements for each group in group lasso
    :return: the loss
    '''
    """
    if stru_param == None:
        return 0.0

    else:
        print(mask_layer.mask)
        print(mask_layer.height)
        print(mask_layer.width)
        # reshape mask to 2D(C,H*W)
        # build index using torch.arrange
        # select pixels using torch.masked_select
        # reshape pixels to (-1,4) size (make sure that it is divisible by 4)
        # compute the norm
        #return mask_layer.mask.view(-1,stru_param).norm(2,dim=1).sum()
        if mask_layer.gl_list == []:
            for i in range(conv_layer.s1*conv_layer.s2):
                save_ker = torch.zeros(1,conv_layer.s1*conv_layer.s2)
                save_ker[0,i] = 1
                save_ker = save_ker.repeat(math.ceil(mask_layer.height/conv_layer.s1), math.ceil(mask_layer.width/conv_layer.s2))[:mask_layer.height,:mask_layer.width].bool()
                save_ker = save_ker.unsqueeze_(0).repeat(mask_layer.channel, 1, 1).cuda()
                mask_layer.gl_list.append(save_ker)

        for kernel in mask_layer.gl_list:
            kernel = torch.masked_select(mask_layer.mask * mask_layer.input,kernel)
            kernel = kernel[:(kernel.shape[0]//stru_param)*stru_param].view(-1,stru_param).norm(2,dim=1).sum().div(128.0)
            gl_list.append(kernel)

        # kernel = (mask_layer.mask * mask_layer.input).view(-1,stru_param).norm(2,dim=1).sum().div_(128.0)
        # gl_list.append(kernel)

def gl_layer(mask_layer, conv_layer, stru_param=4):
    """
    Function for calculating single layer loss.
    '''
    :param mask_layer: mask layer which contains the structure weights
    :param conv_layer: the consecutive conv layer
    :param conv_layer: the number of elements for each group in group lasso
    :return: the loss
    '''
    """
    if stru_param == None:
        return 0.0

    else:
        k1 = conv_layer.k1
        k2 = conv_layer.k2

        all_positions = np.zeros([k1*k2,2]).astype(np.int)

        for i in range(k1):
            for j in range(k2):
                all_positions[i*k2+j,0] = i
                all_positions[i*k2+j,1] = j

        singlelayer_loss = 0
        for i,position in enumerate(all_positions):
            feat_ID = conv_layer.get_feat_ID((position[0],position[1]))
            all_group = array_split(feat_ID,stru_param)
            # print(all_group)
            #print('====================================================')
            single_weight_loss = 0
            for small_group in all_group:#对于每个group
                all_channel_smallgroup_loss = 0
                # print('--------------------------------------------------------')
                for channel in range(mask_layer.channel):#对于每个channel
                    small_group_loss = 0
                    for single in small_group:#对于每个group的element
                        structure_regularization_weight = 0.001
                        # print(mask_layer.mask.shape)
                        # print(channel)
                        #print(single)
                        small_group_loss = small_group_loss + structure_regularization_weight*torch.abs(mask_layer.mask[channel,single[0],single[1]])
                    all_channel_smallgroup_loss = all_channel_smallgroup_loss + small_group_loss
                single_weight_loss = single_weight_loss + all_channel_smallgroup_loss
            singlelayer_loss = singlelayer_loss + single_weight_loss

    return singlelayer_loss

def print_mask(model):
    """
    Function for printing mask value.
    """
    gl_list = []
    conv_layer = None
    mask_layer = None

    for sequence in model.children():
        for sub_sequence in sequence.children():
            if isinstance(sub_sequence, nn.Sequential):
                for layer in sub_sequence.children():
                    if isinstance(layer, Mask):
                        print(layer.mask)

def adjust_gl_rate(rate, epoch):
    if epoch < 100 : rate *= 1
    if 100 <= epoch < 300 : rate *= 2
    return rate
