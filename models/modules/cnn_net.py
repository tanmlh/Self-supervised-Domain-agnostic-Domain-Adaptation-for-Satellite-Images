import torch
import torch.nn as nn
import torch.nn.functional as F
from .base.initialization import initialize
from .base.modules import Conv2dReLU
from . import loss_fun
import pdb
import torchvision as tv
from .utils import check_freeze

def get_cnn_net(net_conf):

    channels = net_conf['channels']
    activation = net_conf['activation'] if 'activation' in net_conf else 'none'
    net = []
    for i in range(len(channels) - 1):
        if i < len(channels) - 2:
            net += [nn.Conv2d(channels[i],
                              channels[i+1], kernel_size=3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(True)]
            # if pool:
            #     net += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            net += [nn.Conv2d(channels[i],
                              channels[i+1], kernel_size=1, stride=1,
                              padding=0, bias=False)]
            if activation == 'sigmoid':
                net += [nn.Sigmoid()]
            elif activation == 'tanh':
                net += [nn.Tanh()]
            elif activation == 'none':
                pass
            elif activation == 'relu':
                net += [nn.ReLU()]
            else:
                raise ValueError

    net = nn.Sequential(*net)
    net.apply(initialize)
    check_freeze(net, net_conf)

    return net

def get_cnn_net_v2(net_conf):
    channels = net_conf['channels']
    norm_type = net_conf['norm_type']
    bias = True if norm_type == 'none' else False
    net = []
    for i in range(len(channels) - 1):
        if i != len(channels) - 2:
            net.append(Conv2dBlock(channels[i], channels[i+1], 3, stride=1, padding=1,
                                   norm=norm_type, activation='lrelu', bias=bias))
        else:
            net.append(Conv2dBlock(channels[i], channels[i+1], 3, stride=1, padding=1,
                                   norm='none', activation='none', bias=bias))

    net = nn.Sequential(*net)
    check_freeze(net, net_conf)
    return net

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride=1,
                 padding=0, dilation=1, norm='none', activation='relu', pad_type='zero', bias=True,
                 use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        # else:
            # assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)
        if use_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class PrivateEncoder(nn.Module):
	def __init__(self, input_channels, code_size):
		super(PrivateEncoder, self).__init__()
		self.input_channels = input_channels
		self.code_size = code_size

		self.cnn = nn.Sequential(nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3), # 128 * 256
								nn.BatchNorm2d(64),
								nn.ReLU(),
								nn.Conv2d(64, 128, 3, stride=2, padding=1), # 64 * 128
								nn.BatchNorm2d(128),
								nn.ReLU(),
								nn.Conv2d(128, 256, 3, stride=2, padding=1), # 32 * 64
								nn.BatchNorm2d(256),
								nn.ReLU(),
								nn.Conv2d(256, 256, 3, stride=2, padding=1), # 16 * 32
								nn.BatchNorm2d(256),
								nn.ReLU(),
								nn.Conv2d(256, 256, 3, stride=2, padding=1), # 8 * 16
								nn.BatchNorm2d(256),
								nn.ReLU())
		self.model = []
		self.model += [self.cnn]
		self.model += [nn.AdaptiveAvgPool2d((1, 1))]
		self.model += [nn.Conv2d(256, code_size, 1, 1, 0)]
		self.model = nn.Sequential(*self.model)

		#self.pooling = nn.AvgPool2d(4)

		#self.fc = nn.Sequential(nn.Conv2d(128, code_size, 1, 1, 0))

	def forward(self, x):
		bs = x.size(0)
		#feats = self.model(x)
		#feats = self.pooling(feats)

		output = self.model(x).view(bs, -1)

		return output
"""
class FourBlocksCNN(nn.Module):
    def __init__(self, net_conf):
        super(FourBlocksCNN, self).__init__()

        self.num_in_channels = net_conf['num_in_channels'] if 'num_in_channels' in net_conf else 3
        self.num_out_channels = net_conf['num_out_channels']
        self.norm_type = net_conf['norm_type'] if 'norm_type' in net_conf else 'in'
        use_sn = net_conf['use_sn'] if 'use_sn' in net_conf else False
        self.use_perc = net_conf['use_perc'] if 'use_perc' in net_conf else False
        strides = net_conf['strides'] if 'strides' in net_conf else [2, 2, 2, 2]
        pads = net_conf['pads'] if 'pads' in net_conf else [2, 1, 1, 1]
        kernels = net_conf['kernels'] if 'kernels' in net_conf else [6, 4, 4, 4]
        temp = 0 if not self.use_perc else 64
        self.feature = nn.Sequential(
            Conv2dBlock(self.num_in_channels, 64, kernels[0], stride=strides[0], padding=pads[0], norm='none',
                        activation='lrelu', bias=False, use_sn=use_sn),
            Conv2dBlock(64, 128, kernels[1], stride=strides[0], padding=pads[1], norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            Conv2dBlock(128, 256, kernels[2], stride=strides[0], padding=pads[2], norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            Conv2dBlock(256, 512, kernels[3], stride=strides[0], padding=pads[3], norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            # nn.Conv2d(512, self.num_out_channels, 1, padding=0),
            # nn.Sigmoid()
        )
        self.cls_net = Conv2dBlock(512, self.num_out_channels, 1, stride=1, padding=0,
                                   norm='none', activation='none', bias=False, use_sn=use_sn)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        if self.use_perc:
            conv_3_3_layer = 1
            vgg19 = tv.models.vgg19(pretrained=True).features
            perceptual_net = nn.Sequential()
            # perceptual_net.add_module('vgg_avg_pool', nn.AvgPool2d(kernel_size=4, stride=4))
            for i,layer in enumerate(list(vgg19)):
                perceptual_net.add_module('vgg_'+str(i),layer)
                if i == conv_3_3_layer:
                    break
            for param in perceptual_net.parameters():
                param.requires_grad = False

            self.perc_net = perceptual_net

        check_freeze(self.feature, net_conf)
        check_freeze(self.cls_net, net_conf)


    def forward(self, x):
        if self.use_perc:
            x2 = self.perc_net(x)
            x = torch.cat([x, x2], dim=1)

        x1 = self.feature(x)
        x1 = self.cls_net(x1)
        x1 = self.global_pooling(x1).view(-1, self.num_out_channels)

        return x1
"""

class FourBlocksCNN(nn.Module):
    def __init__(self, net_conf):
        super(FourBlocksCNN, self).__init__()

        self.norm_type = net_conf['norm_type'] if 'norm_type' in net_conf else 'in'
        use_sn = net_conf['use_sn'] if 'use_sn' in net_conf else False
        channels = net_conf['channels']
        strides = net_conf['strides'] if 'strides' in net_conf else [2, 2, 2, 2]
        pads = net_conf['pads'] if 'pads' in net_conf else [2, 1, 1, 1]
        kernels = net_conf['kernels'] if 'kernels' in net_conf else [6, 4, 4, 4]

        self.feat_1 = Conv2dBlock(channels[0], channels[1], kernels[0], stride=strides[0], padding=pads[0], norm='none',
                                  activation='lrelu', bias=False, use_sn=use_sn)
        self.feat_2 = Conv2dBlock(channels[1], channels[2], kernels[1], stride=strides[1], padding=pads[1], norm=self.norm_type,
                                  activation='lrelu', bias=False, use_sn=use_sn)
        self.feat_3 = Conv2dBlock(channels[2], channels[3], kernels[2], stride=strides[2], padding=pads[2], norm=self.norm_type,
                                  activation='lrelu', bias=False, use_sn=use_sn)
        self.feat_4 =  Conv2dBlock(channels[3], channels[4], kernels[3], stride=strides[3], padding=pads[2], norm=self.norm_type,
                                   activation='lrelu', bias=False, use_sn=use_sn)

        check_freeze(self.feat_1, net_conf)
        check_freeze(self.feat_2, net_conf)
        check_freeze(self.feat_3, net_conf)
        check_freeze(self.feat_4, net_conf)


    def forward(self, x):
        x1 = self.feat_1(x)
        x2 = self.feat_2(x1)
        x3 = self.feat_3(x2)
        x4 = self.feat_4(x3)

        return [x, x1, x2, x3, x4]

class SiameseNet(nn.Module):
    def __init__(self, net_conf):
        super(SiameseNet, self).__init__()

        self.num_out_channels = net_conf['num_out_channels']
        self.norm_type = net_conf['norm_type'] if 'norm_type' in net_conf else 'in'
        self.in_channels = net_conf['num_in_channels']
        use_sn = net_conf['use_sn'] if 'use_sn' in net_conf else False
        self.feature = nn.Sequential(
            Conv2dBlock(self.in_channels, 64, 6, stride=2, padding=2, norm='none', activation='lrelu', bias=False,
                        use_sn=use_sn),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            nn.Conv2d(512, self.num_out_channels, 1, padding=0),
            # nn.Sigmoid()
        )
        self.linear = nn.Linear(self.num_out_channels, 1)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, A, require_prob=False):
        feature_A = self.global_pooling(self.feature(A)).view(-1, self.num_out_channels)
        if require_prob:
            prob_A = self.linear(feature_A)
            return feature_A, prob_A
        return feature_A

    # def forward(self, A, B=None):
    #     if B is not None:
    #         feature_A = self.global_pooling(self.feature(A)).view(-1, self.num_out_channels)
    #         feature_B = self.global_pooling(self.feature(B)).view(-1, self.num_out_channels)
    #         sim_A_B = F.cosine_similarity(feature_A, feature_B, dim=1)
    #         prob_A = self.linear(feature_A)
    #         prob_B = self.linear(feature_B)

    #         return sim_A_B, prob_A, prob_B
    #     else:
    #         out = self.global_pooling(self.feature(A)).view(-1, self.num_out_channels)
    #         out = self.linear(out)

    #         return out

class ListFourBlocksCNN(nn.Module):
    def __init__(self, net_conf):
        super(ListFourBlocksCNN, self).__init__()
        self.num_nets = net_conf['num_doms']
        self.num_out_per_net = net_conf['num_channels']

        temp_conf = {'num_out_channels': self.num_out_per_net}
        self.nets = nn.ModuleList([FourBlocksCNN(temp_conf) for _ in range(self.num_nets)])

        self.paras = nn.Parameter(torch.zeros(self.num_nets, self.num_out_per_net))
        # self.weight = nn.Parameter(torch.ones(self.num_nets, 1))
        self.paras.requires_grad = False

    def forward(self, phase, net_idx=None, x=None):
        if phase == 'train':
            return self.forward_train(x, net_idx)
        elif phase == 'test':
            return self.forward_test(net_idx)
        else:
            raise ValueError('Invalid phase!')

    def forward_train(self, x=None, net_idx=0):
        if x is not None:
            out = self.nets[net_idx](x)
            self.paras[net_idx] = self.paras[net_idx] * 0.95 + out.mean(dim=0)
        else:
            self.paras[net_idx] = self.paras[net_idx] * 0.95 + out.mean(dim=0)

        return out

    def forward_test(self, net_idx):
        if net_idx is not None:
            return self.paras[net_idx]
        else:
            return self.paras.mean(dim=0)

def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        return feat_mean, feat_std

def adain_trans_list(content_feat_list, style_feat_list):

    res = []
    for i, content_feat, style_feat in enumerate(zip(content_feat_list, style_feat_list)):
        if i == len(content_feat_list) - 1:

            assert (content_feat.size()[:2] == style_feat.size()[:2])
            size = content_feat.size()
            style_mean, style_std = calc_mean_std(style_feat)
            content_mean, content_std = calc_mean_std(content_feat)

            normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

            res.append(normalized_feat * style_std.expand(size) + style_mean.expand(size))

    return res

def adain_trans(content_feat, style_feat):

    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain_trans_by_para(feat, para):
    B, C = para.shape
    para_mean = para[:, :C//2].view(B, C//2, 1, 1)
    para_std = para[:, C//2:].view(B, C//2, 1, 1)

    mean, std = calc_mean_std(feat)
    size = feat.size()
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)

    return normalized_feat * para_std + para_mean.expand(size)

def adain_trans_by_para_list(feats, para):
    new_feats = []
    for feat in feats:
        new_feats.append(adain_trans_by_para(feat, para))
    return new_feats

# def adain_trans(feat, params):
#     feat_mean, feat_std = calc_mean_std(feat)

class FeatDiscriminator(nn.Module):
    def __init__(self, net_conf):
        super(FeatDiscriminator, self).__init__()

        self.channels = net_conf['channels']
        self.norm_type = net_conf['norm_type']
        use_sn = net_conf['use_sn'] if 'use_sn' in net_conf else False
        self.feature = nn.Sequential(
            Conv2dBlock(self.channels[0], self.channels[1], 3, stride=1, padding=1, norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            Conv2dBlock(self.channels[1], self.channels[2], 1, stride=1, padding=0, norm=self.norm_type, activation='lrelu',
                        bias=False, use_sn=use_sn),
            Conv2dBlock(self.channels[2], self.channels[3], 1, stride=1, padding=0, norm='none',
                        activation='none', bias=False, use_sn=use_sn),
            # nn.Sigmoid()
        )
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        check_freeze(self.feature, net_conf)

    def forward(self, x):
        x = self.feature(x)
        x = self.global_pooling(x).view(-1, self.channels[-1])
        return x

class DomainNet(nn.Module):
    def __init__(self, net_conf):
        super(DomainNet, self).__init__()
        channels = net_conf['channels']
        norm_type = net_conf['norm_type']
        num_dom = net_conf['num_doms']
        bias = True if norm_type == 'none' else False
        net = []
        for i in range(len(channels) - 2):
            net.append(Conv2dBlock(channels[i], channels[i+1], 3, stride=1, padding=1,
                                   norm=norm_type, activation='lrelu', bias=bias))

        self.channels = channels
        self.weight = nn.Parameter(torch.zeros(num_dom, channels[-2], channels[-1]))
        nn.init.xavier_uniform_(self.weight)

        self.net = nn.Sequential(*net)
        check_freeze(self.net, net_conf)

    def forward(self, x, dom, phase='train'):
        B, C, H, W = x.shape
        x = self.net(x).permute(0, 2, 3, 1).view(B, -1, self.channels[-2])
        cosine = torch.bmm(F.normalize(x), F.normalize(self.weight[dom]))
        cosine = cosine.view(B, H, W, -1).permute(0, 3, 1, 2) * 40

        return cosine, x, self.weight




