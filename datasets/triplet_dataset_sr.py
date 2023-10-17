from __future__ import absolute_import

import os
import io
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from utils.transforms import ImageNetPolicy, RandomDoubleRotate, RandomDoubleFlip
import pdb
from PIL import Image, ImageFilter, ImageOps
import pickle
import time
import random
import tqdm
import itertools
import warnings
from libtiff import TIFF
from skimage.exposure import match_histograms
from .base_dataset import BaseDataset

class TripletDataset(BaseDataset):

    def __init__(self, loader_conf, phase='train'):

        super(TripletDataset, self).__init__(loader_conf, phase)

        self.num_classes = loader_conf['num_classes']
        self.batch_size = loader_conf['batch_size']
        self.patch_size = loader_conf['patch_size']
        self.pad_size = loader_conf['pad_size']
        self.shuffle = loader_conf['shuffle'] if 'shuffle' in loader_conf else False
        self.dom_sample_ratio = loader_conf['dom_sample_ratio'] if 'dom_sample_ratio' in loader_conf else [1, 1]
        self.use_meta = loader_conf['use_meta'] if 'use_meta' in loader_conf else False
        self.meta_val_dom = loader_conf['meta_val_dom'] if self.use_meta else None
        self.meta_test_dom = loader_conf['meta_test_dom'] if 'meta_test_dom' in loader_conf else []
        self.random_scale = loader_conf['random_scale'] if 'random_scale' in loader_conf else 1
        self.filter_size = loader_conf['filter_size'] if 'filter_size' in loader_conf else 1
        self.num_used_data = loader_conf['num_used_data'] if 'num_used_data' in loader_conf else 1e8
        self.dom_A_keys = loader_conf['dom_A_keys'] if 'dom_A_keys' in loader_conf else None
        self.dom_B_keys = loader_conf['dom_B_keys'] if 'dom_B_keys' in loader_conf else None
        self.reverse_gt = loader_conf['reverse_gt'] if 'reverse_gt' in loader_conf else False
        self.aug_conf = loader_conf['aug_conf']
        self.use_hist_equ = loader_conf['use_hist_equ'] if 'use_hist_equ' in loader_conf else False
        self.use_hist_mat = loader_conf['use_hist_mat'] if 'use_hist_mat' in loader_conf else False
        self.sr_dom = loader_conf['sr_dom'] if 'sr_dom' in loader_conf else None
        # self.patch_normalize = loader_conf['patch_normalize'] if 'patch_normalize' in loader_conf else False
        self.norm_type = loader_conf['norm_type'] if 'norm_type' in loader_conf else None
        self.basemap_dom = loader_conf['basemap_dom'] if 'basemap_dom' in loader_conf else None
        self.data_perc = loader_conf['data_perc'] if 'data_perc' in loader_conf else None

        self.dom_data = {}
        self.dom_map = {}
        self.dom_is_train = {}
        dom_cnt = 0

        # augmentation
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = []
        transform.append(transforms.ToTensor())
        transform.append(self.normalize)
        self.transform = transforms.Compose(transform)

        self.imgs = []
        self.gts = []
        with open(self.file_list_path) as f:
            lines = f.readlines()
            self.lines = lines
            tq = tqdm.tqdm(lines)
            for i, line in enumerate(tq):
                tq.set_description('Loading dataset to cpu memory...')
                if i >= self.num_used_data:
                    break

                img_path, gt_path, sr_path, dom_name, is_train = line.strip().split(' ')

                if not dom_name in self.dom_map:
                    self.dom_map[dom_name] = dom_cnt
                    self.dom_data[dom_name] = []
                    self.dom_is_train[dom_name] = int(is_train)
                    dom_cnt += 1

                if self.basemap_dom is not None and dom_name in self.basemap_dom:
                    temp = dom_name + '_basemap'

                    if not temp in self.dom_map:
                        self.dom_map[temp] = dom_cnt
                        self.dom_data[temp] = []
                        self.dom_is_train[temp] = int(is_train)
                        dom_cnt += 1

                    self.dom_data[temp].append((img_path, gt_path, int(is_train)))

                if self.sr_dom is None or dom_name in self.sr_dom:
                    self.dom_data[dom_name].append((sr_path, gt_path, int(is_train)))
                else:
                    self.dom_data[dom_name].append((img_path, gt_path, int(is_train)))

        if self.data_perc is not None:
            for key, perc in self.data_perc.items():
                len_data = len(self.dom_data[key])
                self.dom_data[key] = self.dom_data[key][:int(len_data * perc)]

        self.keys = list(self.dom_data.keys())
        self.lens = [len(self.dom_data[x]) for x in self.keys]
        print('number of domains: {}'.format(dom_cnt))

    def __len__(self):
        return sum(self.lens)

    def get_dom_by_idx(self, idx):
        temp_sum = 0
        for i in range(len(self.lens)):
            temp_sum += self.lens[i]
            if idx < temp_sum:
                return i, idx - temp_sum + self.lens[i]
        raise ValueError('idx is too large!')

    def get_neighbor(self, img_path, gt_path):
        pre, temp = img_path.rsplit('_', 1)
        pre_gt, _ = gt_path.rsplit('_', 1)

        idx, post = temp.split('.')
        row, col = idx.split('x')
        new_idxs = []
        for pos in [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [-1, -1], [1, -1], [-1, 1]]:
        # for pos in [[-2, 0], [2, 0], [0, -2], [0, 2], [2, 2], [-2, -2], [2, -2], [-2, 2]]:

            new_row = int(row) + pos[0]
            new_col = int(col) + pos[1]

            new_path = '{}_{:0>3d}x{:0>3d}.{}'.format(pre, new_row, new_col, post)

            if os.path.exists(os.path.join(self.root, new_path)):
                new_idxs.append([new_row, new_col])

        new_idx = random.choice(new_idxs)
        new_img_path = '{}_{:0>3d}x{:0>3d}.{}'.format(pre, new_idx[0], new_idx[1], post)
        new_gt_path = '{}_{:0>3d}x{:0>3d}.{}'.format(pre_gt, new_idx[0], new_idx[1], 'png')

        return new_img_path, new_gt_path

    def __getitem__(self, idx):

        dom_A, dom_B, idx_A, idx_B = idx

        img_A1_path, gt_A1_path, is_train_A = self.dom_data[dom_A][idx_A]
        img_B1_path, gt_B1_path, is_train_B = self.dom_data[dom_B][idx_B]

        idx_A2 = random.choice(range(self.lens[self.dom_map[dom_A]]))
        img_A2_path, gt_A2_path = self.get_neighbor(img_A1_path, gt_A1_path)


        img_A1 = self.read_img(os.path.join(self.root, img_A1_path))
        img_A2 = self.read_img(os.path.join(self.root, img_A2_path))
        img_B1 = self.read_img(os.path.join(self.root, img_B1_path))

        gt_A1 = self.read_img(os.path.join(self.root, gt_A1_path))
        gt_A2 = self.read_img(os.path.join(self.root, gt_A2_path))
        gt_B1 = self.read_img(os.path.join(self.root, gt_B1_path))

        if self.norm_type == 'patch':
            if self.sr_dom is None or dom_A in self.sr_dom:
                minv_A = min(img_A1.min(), img_A2.min())
                maxv_A = max(img_A1.max(), img_A2.max())
                img_A1 = (img_A1 - minv_A) / (maxv_A - minv_A)
                img_A2 = (img_A2 - minv_A) / (maxv_A - minv_A)

            if self.sr_dom is None or dom_B in self.sr_dom:
                minv_B = img_B1.min()
                maxv_B = img_B1.max()
                img_B1 = (img_B1 - minv_B) / (maxv_B - minv_B)

        elif self.norm_type == 'gray_equ':

            if self.sr_dom is None or dom_A in self.sr_dom:
                minv_A = min(img_A1.min(), img_A2.min())
                maxv_A = max(img_A1.max(), img_A2.max())
                img_A1 = (img_A1 - minv_A) / (maxv_A - minv_A)
                img_A2 = (img_A2 - minv_A) / (maxv_A - minv_A)

                mean_A = (img_A1.mean() + img_A2.mean()) / 2.0
                img_A1 = (img_A1 * 0.5 / mean_A).clip(0, 1)
                img_A2 = (img_A2 * 0.5 / mean_A).clip(0, 1)

            if self.sr_dom is None or dom_B in self.sr_dom:
                minv_B = img_B1.min()
                maxv_B = img_B1.max()
                img_B1 = (img_B1 - minv_B) / (maxv_B - minv_B)

                mean_B = img_B1.mean()
                img_B1 = (img_B1 * 0.5 / mean_B).clip(0, 1)

        elif self.norm_type == 'gray_equ_v2':

            if self.sr_dom is None or dom_A in self.sr_dom:

                mean_A = (img_A1.mean() + img_A2.mean()) / 2.0
                img_A1 = (img_A1 * 0.5 / mean_A)
                img_A2 = (img_A2 * 0.5 / mean_A)

            if self.sr_dom is None or dom_B in self.sr_dom:

                mean_B = img_B1.mean()
                img_B1 = (img_B1 * 0.5 / mean_B)

        elif self.norm_type == 'channel_wise':

            if self.sr_dom is None or dom_A in self.sr_dom:
                for i in range(3):
                    maxv_A = max(img_A1[:, :, i].max(), img_A2[:, :, i].max())
                    img_A1[:, :, i] = img_A1[:, :, i] / maxv_A
                    img_A2[:, :, i] = img_A2[:, :, i] / maxv_A

            if self.sr_dom is None or dom_B in self.sr_dom:
                for i in range(3):
                    maxv_B = img_B1[:, :, i].max()
                    img_B1[:, :, i] = img_B1[:, :, i] / maxv_B

        elif self.norm_type == 'channel_wise_v2':

            if self.sr_dom is None or dom_A in self.sr_dom:
                for i in range(3):
                    mean_A = (img_A1[:, :, i].mean() + img_A2[:, :, i].mean()) / 2
                    img_A1[:, :, i] = img_A1[:, :, i] / mean_A * 0.5
                    img_A2[:, :, i] = img_A2[:, :, i] / mean_A * 0.5

            if self.sr_dom is None or dom_B in self.sr_dom:
                for i in range(3):
                    mean_B = img_B1[:, :, i].mean()
                    img_B1[:, :, i] = img_B1[:, :, i] / mean_B * 0.5

        else:
            raise NotImplementedError()

        if self.use_hist_equ:
            # img_A1 = cv2.equalizeHist((img_A1 * 255).astype(np.uint8))
            # img_A2 = cv2.equalizeHist((img_A2 * 255).astype(np.uint8))
            # img_B1 = cv2.equalizeHist((img_B1 * 255).astype(np.uint8))

            img_A1 = Image.fromarray((img_A1 * 255).astype(np.uint8))
            img_A2 = Image.fromarray((img_A2 * 255).astype(np.uint8))
            img_B1 = Image.fromarray((img_B1 * 255).astype(np.uint8))

            img_A1 = np.array(ImageOps.equalize(img_A1)) / 255.0
            img_A2 = np.array(ImageOps.equalize(img_A2)) / 255.0
            img_B1 = np.array(ImageOps.equalize(img_B1)) / 255.0

        if self.use_hist_mat:
            img_A1 = match_histograms(img_A1, img_B1, multichannel=True)



        img_A1, gt_A1 = self.data_aug(img_A1, gt_A1)
        img_A2, gt_A2 = self.data_aug(img_A2, gt_A2)
        img_B1, gt_B1 = self.data_aug(img_B1, gt_B1)

        if self.reverse_gt:
            gt_A1 = (gt_A1 == 0).long()
            gt_A2 = (gt_A2 == 0).long()
            gt_B1 = (gt_B1 == 0).long()


        out = {}
        out['img_A1'] = img_A1
        out['img_A2'] = img_A2
        out['img_B1'] = img_B1

        out['label_map_A1'] = gt_A1
        out['label_map_A2'] = gt_A2
        out['label_map_B1'] = gt_B1

        out['has_label_A1'] = is_train_A
        out['has_label_A2'] = is_train_A
        out['has_label_B1'] = is_train_B

        return out

    def data_aug(self, img, mask):
        '''
        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        crop_w, crop_h = self.aug_conf['crop_size']

        if self.aug_conf['random_rotate']:
            times = random.choice([0, 1, 2, 3])
            img = np.rot90(img, k=times)
            mask = np.rot90(mask, k=times)

        if self.aug_conf['random_mirror']:
            # random mirror
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

        if self.aug_conf['random_crop']:
            # random scale
            base_w , base_h = self.aug_conf['base_size']
            h, w = img.shape[:2]
            assert w >= h
            if (base_w / w) > (base_h / h):
                base_size = base_w 
                short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                base_size = base_h
                short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
                oh = short_size
                ow = int(1.0 * w * oh / h)

            img = cv2.resize(img, (ow, oh), cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

            # pad crop
            if ow < crop_w or oh < crop_h:
                padh = crop_h - oh if oh < crop_h else 0
                padw = crop_w - ow if ow < crop_w else 0

                img = cv2.copyMakeBorder(img, padh//2, padh - padh//2, padw//2, padw-padw//2,  cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask, padh//2, padh - padh//2, padw//2, padw-padw//2,  cv2.BORDER_REFLECT_101)

            # random crop crop_size
            h, w = img.shape[:2]
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            # img = cv2.crop(img, (x1, y1, x1 + crop_w, y1 + crop_h))
            # mask = cv2.crop(mask, (x1, y1, x1 + crop_w, y1 + crop_h))

            img = img[x1:x1+crop_w, y1:y1+crop_h]
            mask = mask[x1:x1+crop_w, y1:y1+crop_h]

        elif self.aug_conf['resize']:
            img = cv2.resize(img, self.aug_conf['crop_size'], cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.aug_conf['crop_size'], interpolation=cv2.INTER_NEAREST)

        if self.aug_conf['gaussian_blur']:
            # gaussian blur as in PSP
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(
                    radius=random.random()))

        img = self.transform(img.astype(np.float32))
        mask = torch.tensor(mask).long()

        return img, mask

    def read_img(self, file_path):
        if self.use_mc:
            pass
        else:
            if file_path.endswith('tif'):
                img = TIFF.open(file_path, mode='r').read_image()

                if img.shape[-1] == 4:
                    img = img / 65535.0
                    img = img[:, :, [2, 1, 0]]

            else:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        return img

def get_sampler(dataset):
    return IAILSampler(dataset)

class IAILSampler(Sampler):
    def __init__(self, datasets):
        self.phase = datasets.phase
        self.keys = datasets.keys
        self.dom_map = datasets.dom_map
        self.lens = datasets.lens
        self.batch_size = datasets.batch_size
        self.num_iter = len(datasets)
        self.num_doms = len(self.lens)
        self.data_size = sum(self.lens)
        self.img_H = datasets.img_H
        self.img_W = datasets.img_W
        self.pad_size = datasets.pad_size
        self.patch_size = datasets.patch_size
        self.dom_is_train = datasets.dom_is_train
        self.epoch = 0
        self.shuffle = datasets.shuffle
        self.dom_sample_ratio = datasets.dom_sample_ratio
        self.random_scale = datasets.random_scale
        self.dom_A_keys = datasets.dom_A_keys
        self.dom_B_keys = datasets.dom_B_keys

    def __iter__(self):
        random.seed(self.epoch)
        idxes = []

        temp = list(self.dom_is_train.values())
        dom_sample_prob = [self.dom_sample_ratio[0] if x == 1 else self.dom_sample_ratio[1] for x in temp]

        for ite in range(self.num_iter // self.batch_size):
            A_keys = self.dom_A_keys if self.dom_A_keys is not None else self.keys

            # dom_A = random.choices(list(A_keys), weights=dom_sample_prob)[0]
            dom_A = random.choices(list(A_keys))[0]
            idx_A = random.choices(range(self.lens[self.dom_map[dom_A]]), k=self.batch_size)

            # dom_B = random.choices(list(self.keys), weights=dom_sample_prob)[0]
            B_keys = self.dom_B_keys if self.dom_B_keys is not None else self.keys
            dom_B = random.choices(list(B_keys))[0]
            idx_B = random.choices(range(self.lens[self.dom_map[dom_B]]), k=self.batch_size)
            # off_x = random.choices(range(self.img_W), k=self.batch_size)
            # off_y = random.choices(range(self.img_H), k=self.batch_size)
            # off_x, off_y, patch_size = get_roi(self.patch_size, self.img_W, self.random_scale, k=self.batch_size)

            for i in range(self.batch_size):
                idxes += [(dom_A, dom_B, idx_A[i], idx_B[i])]

        self.epoch += 1
        if self.shuffle:
            random.shuffle(idxes)

        return iter(idxes)

    def __len__(self):
        return self.num_iter

