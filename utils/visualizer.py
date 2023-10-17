import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pdb
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:

    def __init__(self, vis_conf = None):
        self.denorm_type = vis_conf['denorm_type'] if 'denorm_type' in vis_conf else None
        self.colors = [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [0, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [  0,   0,   0]] if 'colors' not in vis_conf else vis_conf['colors']

    def visualize(self, vis_type, vis_data):
        if vis_type == 'vis|ent_map':
            return 'image|ent_map', self.get_image_loc_map(*vis_data)
        if vis_type.startswith('vis|bin_mask'):
            return 'image|' + vis_type.split('|')[1], self.get_image_loc_map(*vis_data)
        if vis_type.startswith('vis|images'):
            imgs = []
            for data in vis_data:
                img = self.de_normalize(data)
                img = img.clip(0, 1)
                batch_imgs = self.split_batch(img, 1)
                imgs.append(batch_imgs)
            result = torch.cat(imgs, dim=2)
            return 'image|' + vis_type.split('|')[1], result
        if vis_type.startswith('vis|norm_imgs'):
            imgs = []
            for data in vis_data:
                B, C, H, W = data.shape
                # minv = data.min(dim=3)[0].min(dim=2)[0]
                # maxv = data.max(dim=3)[0].max(dim=2)[0]
                # img = (data - minv.view(B, C, 1, 1)) / (maxv.view(B, C, 1, 1) - minv.view(B, C, 1, 1))
                # img = (data - minv.view(B, C, 1, 1)) / (maxv.view(B, C, 1, 1) - minv.view(B, C, 1, 1))
                # mean = data.mean(dim=[2, 3])
                # std = data.std(dim=[2, 3])
                # img = (data - mean.view(B, C, 1, 1)) / std.view(B, C, 1, 1) * 0.224 + 0.456
                img = data.clip(0, 1)
                batch_imgs = self.split_batch(img, 1)
                imgs.append(batch_imgs)
            result = torch.cat(imgs, dim=2)
            return 'image|' + vis_type.split('|')[1], result

        if vis_type.startswith('vis|pred_maps'):
            maps = []
            num_classes = vis_data[0]
            # img = self.de_normalize(vis_data[1])
            img = vis_data[1]
            img = img.clip(0, 1)
            batch_imgs = self.split_batch(img, 1)

            maps.append(batch_imgs)
            for pred_map in vis_data[2:]:
                cur_map = self.get_image_seg_map(img, pred_map, num_classes, dim=1)
                maps.append(cur_map)

            return 'image|' + vis_type.split('|')[1], torch.cat(maps, dim=2)

        else:
            raise ValueError('No such visualization type!')

    def np2tensor(self, np_img):
        H, W, C = np_img.shape
        img = torch.tensor(np_img).permute([2, 0, 1])
        return img

    def tensor2np(self, torch_img):

        img = torch_img.clip(0, 1).permute([1, 2, 0]).numpy()
        img = img[:, :, [2,1,0]]
        return img

    def de_normalize(self, image):

        if self.denorm_type == 'tensor':
            B, C, H, W = image.shape

            mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1).unsqueeze(0)
            std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1).unsqueeze(0)

            res = image * std + mean
        else:
            res = image + torch.tensor((104.00698793, 116.66876762, 122.67891434)).view(1, 3, 1, 1)
            res /= 255.0

        return res

    def merge_list(self, img_list):
        """
        img_list: list of tensor with shape (C, H, W), where only W varies
        """
        img_W_list = [img.shape[2] for img in img_list]
        img_W = max(img_W_list)
        res_list = []

        for img in img_list:
            if img.shape[2] != img_W:
                pad_img = np.pad(img, ((0, 0), (0, 0), (0, img_W - img.shape[2])), 'constant', constant_values = 0)
                res_list.append(torch.from_numpy(pad_img))
            else:
                res_list.append(img)

        return torch.cat(res_list, dim=1)

    def split_batch(self, image, dim=2):
        """
        Input:
            image: (B, 3, H, W)

        Output:
            tiled_image: (3, B * H, W)
        """
        B, C, H, W = image.shape
        img_list = []
        for i in range(min(B, 16)):
            img_list.append(image[i])
            if i != min(B, 16) - 1:
                if dim == 2:
                    img_list.append(torch.zeros(C, H, 5))
                elif dim == 1:
                    img_list.append(torch.zeros(C, 5, W))

        return torch.cat(img_list, dim=dim)

    def vis_image(self, img):
        # img = self.de_normalize(img)
        img = img.clip(0, 1)
        return self.split_batch(img, 1)

    def get_image_enc_att(self, image, enc_att, downsample=1):
        B, num_heads, label_len, h = enc_att.shape
        B, C, H, W = image.shape

        assert h * downsample == H
        assert C == 3

        w = int(W / downsample)

        att = enc_att.mean(dim=1)
        img = de_normalize(image)

        temp = att.view(B, label_len, 1, h, 1).repeat(1, 1, 3, 1, w).contiguous()
        temp = temp.view(B * label_len, 3, h, w)
        temp = F.interpolate(temp, size=(H, W)).view(B, label_len, 3, H, W)
        temp = temp * 0.2 / temp.mean()

        temp_img = img.view(B, 1, 3, H, W).repeat(1, label_len, 1, 1, 1)


        merged_img = (temp  + temp_img).clamp(0, 1)
        img_list = []
        for i in range(min(B, 16)):
            img_list.append(merged_img[i])

        temp = torch.cat(img_list, dim=2) # (label_len, 3, B * H, W)

        img_list = []
        for i in range(min(label_len, 16)):
            img_list.append(temp[i])

        temp = torch.cat(img_list, dim=2)

        return temp

    def get_image_prob_map(self, image):
        B, label_len, num_classes = image.shape
        minv = image.min()
        maxv = image.max()
        image = (image-minv) / (maxv - minv)
        image = image.unsqueeze(1)

        return split_batch(image)

    def get_image_loc_map(self, image, loc_map, ratio=[0.6, 1], label_upsample=True):
        image = self.de_normalize(image)
        B, C, H, W = image.shape

        if label_upsample is True:
            loc_map = loc_map.unsqueeze(dim=1)
            loc_map = F.interpolate(loc_map.float(), image.shape[2:])
            loc_map = loc_map.squeeze(dim=1)

        temp = loc_map.unsqueeze(1).repeat(1, 3, 1, 1)
        vis_img = torch.clamp((image * ratio[0] + temp * ratio[1]), 0, 1)

        return self.split_batch(vis_img, dim=1)

    def get_false_color_map(self, image, prob_map, ratio=[0.6, 0.4], label_upsample=True, num_color=256):
        cmap = plt.cm.jet(range(num_color))[:, :3] # (num_color, 3)

        image = self.de_normalize(image)
        B, C, H, W = image.shape

        if label_upsample is True:
            prob_map = prob_map.unsqueeze(dim=1)
            prob_map = F.interpolate(prob_map.float(), image.shape[2:])
            prob_map = prob_map.squeeze(dim=1)

        minv, maxv = prob_map.min(), prob_map.max()
        prob_map = (prob_map - minv) / (maxv - minv) * (num_color-1)
        prob_map = prob_map.long()

        prob_map_one_hot = torch.zeros(B, num_color, H, W)
        prob_map_one_hot.scatter_(1, prob_map.unsqueeze(1), torch.ones(B, num_color, H, W))

        temp = prob_map_one_hot.permute([0, 2, 3, 1]).contiguous() # (B, H, W, num_classes)
        temp = temp.view(-1, num_color)
        temp = torch.matmul(temp, torch.from_numpy(cmap).float())
        temp = temp.view(B, H, W, 3)
        temp = temp.permute([0, 3, 1, 2])

        vis_img = (image * ratio[0] + temp * ratio[1])

        return self.split_batch(vis_img, dim=1)

    def get_image_seg_map(self, image, seg_map, num_classes, label_upsample=True, dim=1,
                          ignore_label=None, color_mapper=None, ratio=[0, 1]):
        """
        Input:
            image: (B, 3, H, W)
            seg_map: (B, H, W)

        Output:
            vis_img: (3, B*H, W)
        """
        image = self.de_normalize(image)
        B, C, H, W = image.shape

        if label_upsample is True:
            seg_map = seg_map.unsqueeze(dim=1)
            seg_map = F.interpolate(seg_map.float(), image.shape[2:]).long()
            seg_map = seg_map.squeeze(dim=1)

        if ignore_label is not None:
            seg_map[seg_map==ignore_label] = num_classes
            num_classes += 1

        seg_map_one_hot = torch.zeros(B, num_classes, H, W)
        seg_map_one_hot.scatter_(1, seg_map.unsqueeze(1).long(), torch.ones(B, num_classes, H, W))

        # range_list = list(range(num_classes))
        # random.shuffle(range_list)
        # cmap = plt.cm.Set1(range_list)[:, :3] # (num_classes, 3)
        # if ignore_label is not None:
        #     cmap[-1] = np.array([0, 0, 0])
        if num_classes <= 20:
            cmap = np.array(self.colors) / 255.0
            if color_mapper is not None:
                cmap_copy = np.zeros_like(cmap)
                for key, value in color_mapper.items():
                    cmap_copy[value] = cmap[key]
                cmap = cmap_copy

            cmap = cmap[:num_classes]

        else:
            range_list = list(range(num_classes))
            random.shuffle(range_list)
            cmap = plt.cm.Set1(range_list)[:, :3] # (num_classes, 3)
            if ignore_label is not None:
                cmap[-1] = np.array([0, 0, 0])


        temp = seg_map_one_hot.permute([0, 2, 3, 1]).contiguous() # (B, H, W, num_classes)
        temp = temp.view(-1, num_classes)
        temp = torch.matmul(temp, torch.from_numpy(cmap).float())
        temp = temp.view(B, H, W, 3)
        temp = temp.permute([0, 3, 1, 2])

        vis_img = (image * ratio[0] + temp * ratio[1])


        return self.split_batch(vis_img, dim=dim)
