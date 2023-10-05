import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0, \
              act_type='relu', norm_type='batch', pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)
    act = activation(act_type) if act_type else None
    n = norm(norm_type,out_channels) if norm_type else None
    return sequential(p, conv, n, act)

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm3d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm3d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad3d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad3d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] this module does not support OrderedDict' )
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class DirectionAwareConv(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(DirectionAwareConv, self).__init__()
        
        w = torch.tensor([[[[[0,0,1],
                             [0,1,0], 
                             [1,0,0]]]*3]*in_channels]*(in_channels//4),dtype=torch.float32).cuda()
        h = torch.tensor([[[[[1,0,0],
                             [0,1,0], 
                             [0,0,1]]]*3]*in_channels]*(in_channels//4),dtype=torch.float32).cuda()
        self.k1 = torch.randn((in_channels//4),in_channels,3,3,3).cuda()
        self.k2 = torch.randn((in_channels//4),in_channels,3,3,3).cuda()
        self.kernel_1 = nn.Parameter(torch.mul(self.k1,w))
        self.kernel_2 = nn.Parameter(torch.mul(self.k2,h))

        self.relu  = nn.ReLU(inplace=True)
        self.convx = nn.Conv3d(in_channels, in_channels//4, (1,1,3),1,(0,0,1), bias=bias)
        self.convy = nn.Conv3d(in_channels, in_channels//4, (1,3,1),1,(0,1,0), bias=bias)
        self.convm = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding= 0, bias=bias, act_type='relu')
        
    def forward(self, x):
        xa = self.relu(self.convx(x))
        xb = self.relu(self.convy(x))
        xc = self.relu(F.conv3d(x, self.kernel_1 ,stride=1, padding=(1,1,1)))
        xd = self.relu(F.conv3d(x, self.kernel_2 ,stride=1, padding=(1,1,1)))
        return self.convm(torch.cat((xa,xb,xc,xd),1))


class SparsePyramidConv(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(SparsePyramidConv, self).__init__()

        self.conv1 = ConvBlock(in_channels,    in_channels//4,   kernel_size=3, stride=1, dilation=3,padding=3, bias=bias,act_type='relu')
        self.conv2 = ConvBlock(in_channels,    in_channels//4,   kernel_size=3, stride=1, dilation=2,padding=2, bias=bias,act_type='relu')
        self.conv3 = ConvBlock(in_channels,    in_channels//4,   kernel_size=3, stride=1, dilation=1,padding=1, bias=bias,act_type='relu')
        self.conv4 = ConvBlock((in_channels//4)*3, out_channels, kernel_size=1, stride=1, dilation=1,padding=0, bias=bias,act_type=None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.conv4(torch.cat((x1, x2, x3), 1))

class StripeDirectionBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(StripeDirectionBlock, self).__init__()
        self.directionawareconv = DirectionAwareConv(in_channels,out_channels,bias)
    def forward(self, x):
        return self.directionawareconv(x)

class StripeSparseBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(StripeSparseBlock, self).__init__()
        self.sparsepyramidconv  = SparsePyramidConv(in_channels,out_channels,bias)
    def forward(self, x):
        return self.sparsepyramidconv(x)
        
class StripeAttributeAwareBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(StripeAttributeAwareBlock, self).__init__()
        
        self.directionawareconv = DirectionAwareConv(in_channels,in_channels,bias)
        #self.directionawareconv = DirectionAwareConv(in_channels,out_channels,bias)
        self.sparsepyramidconv  = SparsePyramidConv(in_channels,out_channels,bias)

    def forward(self, x):
        x_ = self.directionawareconv(x)
        y  = self.sparsepyramidconv(x_)
        return y#self.directionawareconv(x)#y#self.sparsepyramidconv(x)#self.directionawareconv(x) #y#self.sparsepyramidconv(x)#y

class SandwichConv(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(SandwichConv, self).__init__()

        self.conv = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels//2,(1,3,3),1,(0,1,1), bias=bias),
                        nn.ReLU(True),
                        nn.Conv3d(out_channels//2, out_channels,(1,3,3),1,(0,1,1), bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = ConvBlock(nc, gc, kernel_size=3, stride=1, padding=1, bias=bias, act_type='relu')
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size=3, stride=1, padding=1, bias=bias, act_type='relu')
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size=3, stride=1, padding=1, bias=bias, act_type='relu')
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size=3, stride=1, padding=1, bias=bias, act_type='relu')
        self.conv5 = ConvBlock(nc+4*gc, nc, kernel_size=3, stride=1, padding=1, bias=bias, act_type=None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x
