import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision as tv
from . import cnn_net


class GANLoss(nn.Module):

    def __init__(self, net_conf):
        super(GANLoss, self).__init__()
        self.net_conf = net_conf
        self.dis_net = cnn.get_cnn_net(net_conf)
        self.l1_loss_fun = nn.SmoothL1Loss()

    def forward(self, pred_pos, tensors, loss_type='gen'):

        label = tensors['label']
        B, label_len = label.shape
        label_mask = (label > 0).float()
        masked_pred_pos = pred_pos * label_mask.unsqueeze(2) # (B, label_len, 2)

        if loss_type == 'gen':
            D_fake = self.dis_net(pred_pos.view(B, -1))
            tensors['gan_real_label'].resize_(D_fake.size()).fill_(1)
            loss_GAN = self.l1_loss_fun(D_fake, tensors['gan_real_label'])

        else:

            real_pos = tensors['cen_pos']
            # masked_real_pos = real_pos[label_mask[:, 0], label_mask[:, 1], :]
            masked_real_pos = real_pos * label_mask.unsqueeze(2)
            D_fake = self.dis_net(masked_pred_pos.view(B, -1))
            D_real = self.dis_net(masked_real_pos.view(B, -1))

            tensors['gan_fake_label'].resize_(D_fake.size()).fill_(0)
            tensors['gan_real_label'].resize_(D_fake.size()).fill_(1)

            loss_D1 = self.l1_loss_fun(D_fake, tensors['gan_fake_label'])
            loss_D2 = self.l1_loss_fun(D_real, tensors['gan_real_label'])
            loss_GAN = (loss_D1 + loss_D2) / 2

        return loss_GAN * 0.01

class PerceptualLoss(nn.Module):
    def __init__(self, net_conf):
        super(PerceptualLoss, self).__init__()

        conv_3_3_layer = 14
        vgg19 = tv.models.vgg19(pretrained=True).features
        perceptual_net = nn.Sequential()
        perceptual_net.add_module('vgg_avg_pool', nn.AvgPool2d(kernel_size=4, stride=4))
        for i,layer in enumerate(list(vgg19)):
            perceptual_net.add_module('vgg_'+str(i),layer)
            if i == conv_3_3_layer:
                break
        for param in perceptual_net.parameters():
            param.requires_grad = False

        self.net = perceptual_net
        self.mse_loss_fun = nn.MSELoss()


    def forward(self, img_A, img_B=None):
        if img_B is not None:
            return self.mse_loss_fun(self.net(img_A), self.net(img_B))
        return self.net(img_A)

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        out_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(out_grad, gt_grad), out_grad, gt_grad

def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        return feat_mean, feat_std

class NormalizedL1Loss(nn.Module):

    def __init__(self, net_conf):
        super(NormalizedL1Loss, self).__init__()
        self.loss_fun = nn.L1Loss()
        self.stages = net_conf['loss_cons_stages'] if 'loss_cons_stages' in net_conf else [-1]

    def forward(self, feats1, feats2):

        loss = 0
        for stage in self.stages:

            feat1 = feats1[stage]
            feat2 = feats2[stage]

            size = feat1.size()

            mean1, std1 = calc_mean_std(feat1)
            mean2, std2 = calc_mean_std(feat2)

            normalized1 = (feat1 - mean1.expand(size)) / std1.expand(size)
            normalized2 = (feat2 - mean2.expand(size)) / std2.expand(size)

            loss += self.loss_fun(normalized1, normalized2)

        return loss

class FeatureL1Loss(nn.Module):

    def __init__(self, net_conf):
        super(FeatureL1Loss, self).__init__()
        self.loss_fun = nn.L1Loss()
        self.stages = net_conf['loss_cons_stages'] if 'loss_cons_stages' in net_conf else [-1]

    def forward(self, feats1, feats2):

        loss = 0
        for stage in self.stages:

            feat1 = feats1[stage]
            feat2 = feats2[stage]

            loss += self.loss_fun(feat1, feat2)

        return loss
