import torch
import torch.nn as nn
from .unet.model import Unet
from .fpn.model import FPN
from .transformer.Models import Transformer
import torch.nn.functional as F
import pdb
from .encoders import get_encoder
from .utils import check_freeze
from .cnn_net import FourBlocksCNN

def get_enc_net(net_conf):

    if net_conf['name'] == 'four_blocks':
        net = FourBlocksCNN(net_conf)
    else:
        net = get_encoder(net_conf['name'],
                          in_channels=net_conf['num_in_channels'],
                          depth=net_conf['depth'],
                          weights='imagenet',
        )

    check_freeze(net, net_conf)
    return net


def get_enc_dec_net(net_conf):
    enc_conf = net_conf['enc_net']
    if enc_conf['net_name'] == 'fpn_resnet_50':
        net = FPN('resnet50',
                  in_channels=enc_conf['num_in_channels'],
                  classes=enc_conf['num_out_channels'],
                  encoder_depth=enc_conf['depth'],
                  upsampling=enc_conf['upsampling'])
    elif enc_conf['net_name'] == 'fpn_vgg_16':
        net = FPN('vgg16_bn',
                  in_channels=enc_conf['num_in_channels'],
                  classes=enc_conf['num_out_channels'],
                  encoder_depth=enc_conf['depth'],
                  upsampling=enc_conf['upsampling'])
    elif enc_conf['net_name'].startswith('unet_resnet') == 'unet_resnet_50':
        net = Unet('resnet{}'.format(enc_conf['net_name'].split('_')[-1]),
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   use_y_avg_pool=enc_conf['use_y_avg_pool'],
                   use_x_avg_pool=enc_conf['use_x_avg_pool'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'unet_darts':
        net = Unet('darts_imagenet',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   use_y_avg_pool=enc_conf['use_y_avg_pool'],
                   use_x_avg_pool=enc_conf['use_x_avg_pool'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'unet_resnet_101':
        net = Unet('resnet101',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   use_y_avg_pool=enc_conf['use_y_avg_pool'],
                   use_x_avg_pool=enc_conf['use_x_avg_pool'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'unet_vgg_16':
        net = Unet('vgg16_bn',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'fpn_resnet_50_3x':
        net = FPN('resnet50',
                  in_channels=enc_conf['num_in_channels'],
                  classes=enc_conf['num_out_channels'],
                  encoder_depth=enc_conf['depth'],
                  upsampling=enc_conf['upsampling'])
    elif enc_conf['net_name'] == 'resnet_50_4x':
        net = resnet_v1.resnet50(True)

    elif enc_conf['net_name'] == 'unet_efficient_net':
        net = Unet('efficientnet-b3',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   decoder_attention_type='scse')


    else:
        raise ValueError

    check_freeze(net, enc_conf)
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

# class FourBlocksCNN(nn.Module):
#     def __init__(self, net_conf):
#         super(FourBlocksCNN, self).__init__()
# 
#         self.norm_type = net_conf['norm_type'] if 'norm_type' in net_conf else 'in'
#         use_sn = net_conf['use_sn'] if 'use_sn' in net_conf else False
#         channels = net_conf['channels']
# 
#         self.feat_1 = Conv2dBlock(channels[0], channels[1], 6, stride=2, padding=2, norm='none',
#                                   activation='lrelu', bias=False, use_sn=use_sn)
#         self.feat_2 = Conv2dBlock(channels[1], channels[2], 4, stride=2, padding=1, norm=self.norm_type,
#                                   activation='lrelu', bias=False, use_sn=use_sn)
#         self.feat_3 = Conv2dBlock(channels[2], channels[3], 4, stride=2, padding=1, norm=self.norm_type,
#                                   activation='lrelu', bias=False, use_sn=use_sn)
#         self.feat_4 =  Conv2dBlock(channels[3], channels[4], 4, stride=2, padding=1, norm=self.norm_type,
#                                    activation='lrelu', bias=False, use_sn=use_sn)
# 
#         check_freeze(self.feat_1, net_conf)
#         check_freeze(self.feat_2, net_conf)
#         check_freeze(self.feat_3, net_conf)
#         check_freeze(self.feat_4, net_conf)
# 
# 
#     def forward(self, x):
#         x1 = self.feat_1(x)
#         x2 = self.feat_2(x1)
#         x3 = self.feat_3(x2)
#         x4 = self.feat_4(x3)
# 
#         return [x, x1, x2, x3, x4]
