import random
import numpy as np
import pdb
import operator
import os
import sys
import time
import tqdm
import pickle
import itertools
from PIL import Image
import random
import copy

import torch
import torchvision as tv
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from models.base_solver import BaseSolver
from models.modules import enc_net, dec_net, cnn_net, dnn_net
from models.modules import loss_fun
from utils import visualizer
from utils import metric
from models import tester

def get_solver(conf):
    return MulDASolver(conf)

def get_model(conf):
    return MulDAModel(conf)

class MulDASolver(BaseSolver):

    def init_tensors(self):
        pass

    def set_tensors(self, batch):
        self.tensors = {}
        for key, value in batch.items():
            if type(value) == torch.Tensor:
                self.tensors[key] = value.cuda()

    def process_batch(self, batch, phase, record=True):
        if phase == 'train':
            return self.train_batch(batch)
        elif phase == 'test':
            return self.test_batch(batch, record)
        else:
            raise ValueError('Invalid phase name!')

    def train_batch(self, batch):

        self.set_tensors(batch)
        self.net.train()
        dis_step_freq = self.net_conf['dis_step_freq']
        print_test_iou = self.solver_conf['print_test_iou'] if 'print_test_iou' in self.net_conf else True

        if self.global_step % 1 == 0:
            loss, state = self.net.forward(self.tensors, 'train_G')
            self.net.zero_grad()
            self.my_backward(loss)
            self.optimizers['gen'].step()

            self.tensors['img_A2B'] = state['else|img_A2B']

        if self.global_step % dis_step_freq == 0:
            loss, temp = self.net.forward(self.tensors, 'train_D')
            self.net.zero_grad()
            self.my_backward(loss)
            self.optimizers['dis'].step()

            for key, value in temp.items():
                state[key] = value

        for key, value in state.items():
            if key.split('|')[0] == 'scalar':
                state[key] = value.mean().cpu().item()

        return state

class MulDAModel(nn.Module):
    def __init__(self, net_conf):
        super(MulDAModel, self).__init__()

        self.net_conf = net_conf
        self.phase = net_conf['phase'] if 'phase' in net_conf else 'train'

        self.reg_enc_net = enc_net.get_enc_net(net_conf['reg_enc_net'])
        self.reg_dec_net = dec_net.get_dec_net(net_conf['reg_dec_net'])
        self.reg_net = cnn_net.get_cnn_net(net_conf['reg_net'])
        self.use_rgb = net_conf['use_rgb'] if 'use_rgb' in net_conf else False

        if self.phase == 'train':
            self.dis_net = cnn_net.SiameseNet(net_conf['dis_net'])

        self.mse_loss_fun = nn.MSELoss()
        self.l1_loss_fun = nn.SmoothL1Loss()
        self.cse_loss_fun = nn.CrossEntropyLoss()
        self.nll_loss_fun = nn.NLLLoss()
        self.perc_loss_fun = loss_fun.PerceptualLoss(net_conf)

        self.margin = net_conf['margin'] if 'margin' in net_conf else 0.3


    def parameters_group(self):
        reg_enc_para = self.reg_enc_net.parameters()
        reg_dec_para = self.reg_dec_net.parameters()
        reg_para = self.reg_net.parameters()
        dis_para = self.dis_net.parameters()

        para_grp_1 = itertools.chain(reg_enc_para, reg_dec_para, reg_para)
        para_grp_2 = dis_para

        return {'gen': para_grp_1, 'dis': para_grp_2}

    def forward(self, tensors, phase):

        if phase == 'train_G':
            return self.forward_G(tensors)
        elif phase == 'train_D':
            return self.forward_D(tensors)
        elif phase == 'test':
            return self.inference(tensors)
        elif phase == 'dis_net':
            return self.forward_dis_net(tensors)
        else:
            raise ValueError

    def forward_G(self, tensors):

        state = {}

        img_A1 = tensors['img_A1'] # (B, 3, H, W)
        img_A2 = tensors['img_A2'] # (B, 3, H, W)
        img_B1 = tensors['img_B1'] # (B, 3, H, W)

        if self.use_rgb:
            img_A1 = img_A1[:, :3]
            img_A2 = img_A2[:, :3]
            img_B1 = img_B1[:, :3]

        feats_A = self.reg_enc_net(img_A1) # [(B, C, h, w), ...]
        feats_B = self.reg_enc_net(img_B1) # [(B, C, h, w), ...]

        B, C, H, W = feats_A[-1].shape

        feats_A2B = cnn_net.adain_trans(feats_A[-1], feats_B[-1])
        feats_B2A = cnn_net.adain_trans(feats_B[-1], feats_A[-1])

        feats_A2B = feats_A[:-1] + [feats_A2B]
        feats_B2A = feats_B[:-1] + [feats_B2A]

        rec_img_A = self.reg_net(self.reg_dec_net(feats_A))
        rec_img_B = self.reg_net(self.reg_dec_net(feats_B))

        img_A2B = self.reg_net(self.reg_dec_net(feats_A2B))
        img_B2A = self.reg_net(self.reg_dec_net(feats_B2A))

        feats_A2B_ = self.reg_enc_net(img_A2B)
        feats_B2A_ = self.reg_enc_net(img_B2A)

        feats_A2B2A = cnn_net.adain_trans(feats_A2B_[-1], feats_B2A_[-1])
        feats_B2A2B = cnn_net.adain_trans(feats_B2A_[-1], feats_A2B_[-1])

        feats_A2B2A = feats_A2B_[:-1] + [feats_A2B2A]
        feats_B2A2B = feats_B2A_[:-1] + [feats_B2A2B]

        rec_img_A2B2A = self.reg_net(self.reg_dec_net(feats_A2B2A))
        rec_img_B2A2B = self.reg_net(self.reg_dec_net(feats_B2A2B))

        """ Loss Calculation """
        loss_total = None

        loss_w = self.net_conf['loss_weight_G']

        """ Self-reconstruction loss """
        loss_self = self.l1_loss_fun(img_A1, rec_img_A) + self.l1_loss_fun(img_B1, rec_img_B)

        """ Cross-reconstruction loss """
        loss_cross = self.l1_loss_fun(img_A1, rec_img_A2B2A) + self.l1_loss_fun(img_B1, rec_img_B2A2B)

        """ Perceptual loss """
        if self.use_rgb:
            loss_perc = self.perc_loss_fun(img_A1, rec_img_A2B2A)
            loss_perc += self.perc_loss_fun(img_A1, rec_img_A)
            loss_perc += self.perc_loss_fun(img_B1, rec_img_B2A2B)
            loss_perc += self.perc_loss_fun(img_B1, rec_img_B)

        """ Outlier loss """
        dis_feat_A1 = self.dis_net(img_A1)
        dis_feat_A2 = self.dis_net(img_A2)
        dis_feat_B1 = self.dis_net(img_B1)
        dis_feat_A2B, pred_prob_A2B = self.dis_net(img_A2B, require_prob=True)

        loss_gan = torch.mean((pred_prob_A2B - torch.zeros(B,).cuda()) ** 2)
        state['scalar|loss_ls_gan_G'] = loss_gan.detach()

        """ InfoNCE Loss """
        # feats_X1 = F.normalize(dis_feat_A1, dim=1)
        feats_X2 = F.normalize(dis_feat_A2B, dim=1)
        feats_Y = F.normalize(torch.cat([dis_feat_A2, dis_feat_B1]), dim=1)
        # mat_X1_Y = torch.matmul(feats_X1, feats_Y.T) # (B, 2*B)
        mat_X2_Y = torch.matmul(feats_X2, feats_Y.T) # (B, 2*B)

        label_X2_Y = torch.arange(0, B).cuda() + B
        loss_nce = self.cse_loss_fun(mat_X2_Y.view(B, -1), label_X2_Y)



        """ Total loss """
        loss_total = sum([x * loss_w[i] for i, x in enumerate([loss_self, loss_cross,
                                                               loss_perc if self.use_rgb else 0,
                                                               loss_gan, loss_nce])])
        state['scalar|loss'] = loss_total.detach()


        to_cpu = lambda x: x.detach().cpu()
        state['else|rec_img_A'] = rec_img_A.detach()
        state['else|rec_img_B'] = rec_img_B.detach()
        state['else|img_A2B'] = img_A2B.detach()
        state['else|img_B2A'] = img_B2A.detach()
        state['else|img_A2B2A'] = rec_img_A2B2A.detach()
        state['else|img_B2A2B'] = rec_img_B2A2B.detach()
        state['scalar|loss_self'] = loss_self.detach()
        state['scalar|loss_cros'] = loss_cross.detach()
        state['scalar|loss_nce_G'] = loss_nce.detach()
        state['scalar|loss_gan_G'] = loss_gan.detach()
        state['vis|norm_imgs_A1_A2_B_A2B_B2A'] = map(to_cpu, [img_A1, img_A2, img_B1, img_A2B, img_B2A])
        state['vis|norm_imgs_A1_rec_A1'] = map(to_cpu, [img_A1, rec_img_A])

        return loss_total, state

    def forward_D(self, tensors):

        state = {}

        img_A1 = tensors['img_A1']
        img_A2 = tensors['img_A2']
        img_B1 = tensors['img_B1']
        img_A2B = tensors['img_A2B']

        B, C, H, W = img_A1.shape

        if self.use_rgb:
            img_A1 = img_A1[:, :3]
            img_A2 = img_A2[:, :3]
            img_B1 = img_B1[:, :3]

        dis_feat_A1, pred_prob_A1 = self.dis_net(img_A1, require_prob=True)
        dis_feat_A2, pred_prob_A2 = self.dis_net(img_A2, require_prob=True)
        dis_feat_B1, pred_prob_B1 = self.dis_net(img_B1, require_prob=True)
        dis_feat_A2B, pred_prob_A2B = self.dis_net(img_A2B, require_prob=True)

        """ GAN Loss """
        loss_gan = torch.mean((pred_prob_A1 - torch.zeros(B,).cuda()) ** 2)
        loss_gan += torch.mean((pred_prob_A2 - torch.zeros(B,).cuda()) ** 2)
        loss_gan += torch.mean((pred_prob_B1 - torch.zeros(B,).cuda()) ** 2)
        loss_gan += torch.mean((pred_prob_A2B - torch.ones(B,).cuda()) ** 2)

        """ InfoNCE Loss """
        feats_X1 = F.normalize(dis_feat_A1, dim=1)
        feats_Y = F.normalize(torch.cat([dis_feat_A2, dis_feat_A2B]), dim=1)

        mat_X1_Y = torch.matmul(feats_X1, feats_Y.T) # (B, 2*B)

        label_X1_Y = torch.arange(0, B).cuda()

        loss_nce = self.cse_loss_fun(mat_X1_Y, label_X1_Y)
        # loss_nce += self.cse_loss_fun(mat_X2_Y, label_X2_Y)
        # loss_nce /= 2

        loss_w = self.net_conf['loss_weight_D']
        loss_total = sum([x * loss_w[i] for i, x in enumerate([loss_gan, loss_nce])])

        state['scalar|loss_gan_D'] = loss_gan.detach()
        state['scalar|loss_nce_D'] = loss_nce.detach()

        return loss_total, state

    def inference(self, tensors):

        state = {}

        img_A = tensors['img_A1'] # (B, 3, H, W)
        img_A2 = tensors['img_A2']
        img_B = tensors['img_B1'] if 'img_B1' in tensors else None
        B = img_A.shape[0]

        feats_A = self.reg_enc_net(img_A) # [(B, C, h, w), ...]
        if img_B is not None:
            feats_B = self.reg_enc_net(img_B) # [(B, C, h, w), ...]
            feats_A2B = cnn_net.adain_trans(feats_A[-1], feats_B[-1])
            feats_A2B = feats_A[:-1] + [feats_A2B]

            feats_B2A = cnn_net.adain_trans(feats_B[-1], feats_A[-1])
            feats_B2A = feats_B[:-1] + [feats_B2A]

            img_A2B = self.reg_net(self.reg_dec_net(feats_A2B))
            img_B2A = self.reg_net(self.reg_dec_net(feats_B2A))
        else:
            img_A2B = img_A

        to_cpu = lambda x: x.detach().cpu()
        state['else|feat_A'] = feats_A[-1].detach()
        state['else|img_A2B'] = img_A2B.detach()
        state['save|vis_imgs'] = map(to_cpu, [img_A[0], img_A2[0], img_B[0], img_A2B[0], img_B2A[0]])

        return state

    def forward_dis_net(self, tensors):
        img_A = tensors['img_A1'] # (B, 3, H, W)
        img_B = tensors['img_B1']

        feat_A, prob_A = self.dis_net(img_A, require_prob=True)
        feat_B, prob_B = self.dis_net(img_B, require_prob=True)


        sim_A_B = F.cosine_similarity(feat_A, feat_B, dim=1)

        return sim_A_B
