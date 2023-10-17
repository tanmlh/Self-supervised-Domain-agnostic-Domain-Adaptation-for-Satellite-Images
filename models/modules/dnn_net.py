import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .base.initialization import initialize

def get_dnn_net(net_conf):

    channels = net_conf['channels']
    norm_type = net_conf['norm_type']

    net = []
    # net.append(nn.BatchNorm1d(channels[0]))
    for i in range(len(channels)-1):
        net.append(nn.Linear(channels[i], channels[i+1]))

        if norm_type == 'bn':
            net.append(nn.BatchNorm1d(channels[i+1]))
        elif norm_type == 'none':
            pass
        else:
            raise ValueError

        if i != len(channels)-2:
            net.append(nn.ReLU(True))

    net = nn.Sequential(*net)
    net.apply(initialize)

    return net

class AdaInPara(nn.Module):
    def __init__(self, net_conf):
        super(AdaInPara, self).__init__()
        num_doms = net_conf['num_doms']
        num_channels = net_conf['num_channels']
        self.paras = nn.Parameter(torch.zeros(num_doms, num_channels))

    def forward(self, dom_idx):
        return self.paras[dom_idx]

class AdaInParaV2(nn.Module):
    def __init__(self, net_conf):
        super(AdaInParaV2, self).__init__()
        num_doms = net_conf['num_doms']
        num_channels = net_conf['num_channels']
        self.paras = nn.Parameter(torch.zeros(num_doms, num_channels))
        self.weight = nn.Parameter(torch.ones(num_doms, 1))
        freeze = True if 'freeze' in net_conf and net_conf['freeze'] is True else False
        if freeze:
            self.paras.requires_grad = False
            self.weight.requires_grad = False

    def forward(self, dom_idx=None):
        if dom_idx is not None:
            x = self.paras[dom_idx]
        else:
            x = self.paras
            weight = F.sigmoid(self.weight)
            weight = weight / weight.sum(dim=0)
            x = (x * weight).sum(dim=0).unsqueeze(0)
        # elif dom_idx == -1:
        #     return self.paras.mean(dim=0)

        return x

def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        return feat_mean, feat_std

def calc_mean_std2(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C)

    return torch.cat([feat_mean, feat_std], dim=1)

def adain_trans_by_para(feat, para):
    B, C = para.shape
    para_mean = para[:, :C//2].view(B, C//2, 1, 1)
    para_std = para[:, C//2:].view(B, C//2, 1, 1)

    mean, std = calc_mean_std(feat)
    size = feat.size()
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)

    return normalized_feat * para_std + para_mean.expand(size)

class Adaptator(nn.Module):
    def __init__(self, net_conf):
        super(Adaptator, self).__init__()
        num_channels = net_conf['num_channels']
        self.nets = nn.ModuleList([nn.Linear(channel*2, channel*2) for channel in num_channels])

    def forward(self, feats):
        ada_params = [self.nets[i](calc_mean_std2(x)) for (i, x) in enumerate(feats[1:])]
        ada_feats = [adain_trans_by_para(feat, para) for feat, para in zip(feats[1:], ada_params)]

        return [feats[0]] + ada_feats

class ColorMap(nn.Module):
    def __init__(self, net_conf):
        super(ColorMap, self).__init__()
        self.w = nn.Parameter(torch.ones(256, 256, 256))
        self.k = nn.Parameter(torch.zeros(256, 256, 256))

    def forward(img):
        B, C, H, W = img.shape
        assert C == 3

        idx = img[:, 0] * 256 * 256 + img[:, 1] * 256 + img[:, 2]
        w_flat = self.w.view(-1)
        k_flat = self.k.view(-1)
        gen_img = w_flat[idx.view(-1)].view(B, 1, H, W) * img + k_flat[idx].view(B, 1, H, W)

        return gen_img
