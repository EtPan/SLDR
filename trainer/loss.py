import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mydistance(x,y):
    F = x-y
    g = torch.sqrt(torch.pow(F, 2) + 1e-6)
    mg = g.sum(4, keepdim=True).sum(3, keepdim=True).sum(2,keepdim=True)/ (g.size(2)*g.size(3)*g.size(4))
    return mg.squeeze(4).squeeze(3).squeeze(2)

def cal_sam(X,Y):
    esp = 1e-6
    InnerPro = torch.sum(X*Y,1,keepdim=True)
    len1 = torch.norm(X,p=2,dim=1,keepdim=True)
    len2 = torch.norm(Y,p=2,dim=1,keepdim=True)
    divisor = len1*len2
    mask = torch.eq(divisor,0)
    divisor = divisor + (mask.float())*esp
    cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp,1-esp)
    sam = torch.acos(cosA)
    return torch.mean(sam)/np.pi

class WTVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(WTVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x, mask):
        
        batch_size = x.size()[0]
        
        count_x = self._tensor_size(x)
        h_x = x.size()[-1]
        w_x = x.size()[-2]
        epsilon = 1e-3
        mask = (mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
        alpha = mask + epsilon

        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])

        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()

        tv = (h_tv / count_h + w_tv / count_w) / batch_size
        weight = alpha.sum()/(count_x*batch_size)

        return self.TVLoss_weight * tv * weight.cuda()

    def _tensor_size(self, t):
        return t.size()[-1] * t.size()[-2] * t.size()[-3]


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[-1]
        w_x = x.size()[-2]
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[-1] * t.size()[-2] * t.size()[-3]

class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[-3]
        count_c = self._tensor_size(x[:, :, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[-1] * t.size()[-2] * t.size()[-3]
