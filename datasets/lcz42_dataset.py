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
import h5py
from .base_dataset import BaseDataset
from datasets import data_augmentor, data_normalizer, data_reader

class LCZ42Dataset(Dataset):

    def __init__(self, loader_conf, phase='train'):

        super(LCZ42Dataset, self).__init__()

        self.root = loader_conf['dataset_dir']
        self.train_file_path = loader_conf['train_file_path']
        self.test_file_path = loader_conf['test_file_path'] if 'test_file_path' in loader_conf else None
        self.img_H = loader_conf['img_H']
        self.img_W = loader_conf['img_W']
        self.phase = phase
        self.use_resize = loader_conf['use_resize'] if 'use_resize' in loader_conf else True
        self.num_used_data = loader_conf['num_used_data'] if 'num_used_data' in loader_conf else 1e8
        self.aug_conf = loader_conf['aug_conf']
        self.use_hist_equ = loader_conf['use_hist_equ'] if 'use_hist_equ' in loader_conf else False
        self.use_hist_mat = loader_conf['use_hist_mat'] if 'use_hist_mat' in loader_conf else False
        self.use_rgb = loader_conf['use_rgb'] if 'use_rgb' in loader_conf else False
        self.norm_type = loader_conf['norm_type'] if 'norm_type' in loader_conf else None
        self.A_sample_type = loader_conf['A_sample_type']
        self.B_sample_type = loader_conf['B_sample_type']

        self.normalizer = data_normalizer.DataNormalizer(loader_conf)
        self.augmentor = data_augmentor.DataAugmentor(self.aug_conf)

        transform = []
        transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

        self.imgs = []
        self.gts = []

        f1 = h5py.File(self.train_file_path, 'r')
        self.data1 = f1['sen2']
        self.labels1 = f1['label']
        self.len1 = len(self.data1)

        f2 = h5py.File(self.test_file_path, 'r')
        self.data2 = f2['sen2']
        self.labels2 = f2['label']
        self.len2 = len(self.data2)

        if self.use_hist_equ:
            data_list = itertools.chain(self.data1, self.data2)
            self.normalizer.fit_hist(data_list)

    def __len__(self):
        return self.len1 if self.phase == 'train' else self.len2

    def get_sample(self, sample_type):
        if sample_type == 'random':
            flag = random.random() < 0.5
        elif sample_type == 'data1':
            flag = 0
        else:
            flag = 1

        data = self.data1 if flag == 0 else self.data2
        labels = self.labels1 if flag == 0 else self.labels2
        length = self.len1 if flag == 0 else self.len2

        idx = random.choice(range(length))
        cur_img = data[idx]
        cur_label = labels[idx]

        return cur_img, cur_label

    def __getitem__(self, idx):

        if self.phase == 'train':
            img_A, gt_A = self.get_sample(self.A_sample_type)
            img_B, gt_B = self.get_sample(self.B_sample_type)
        else:
            img_A, gt_A = self.data2[idx], self.labels2[idx]
            img_B, gt_B = img_A, gt_A

        img_A  = self.normalizer.normalize(img_A)
        img_B  = self.normalizer.normalize(img_B)

        if self.use_hist_equ:
            img_A = self.normalizer.hist_equalize(img_A)
            img_B = self.normalizer.hist_equalize(img_B)

        elif self.use_hist_mat:
            img_A = self.normalizer.hist_match(img_A, img_B)

        if self.phase == 'train':
            img_A1, _ = self.augmentor.data_aug(img_A)
            img_A2, _ = self.augmentor.data_aug(img_A)
            img_B1, _ = self.augmentor.data_aug(img_B)
        else:
            img_A1 = img_A
            img_A2 = img_A
            img_B1 = img_B

        img_A1 = self.transform(img_A1.astype(np.float32))
        img_A2 = self.transform(img_A2.astype(np.float32))
        img_B1 = self.transform(img_B1.astype(np.float32))

        gt_A = torch.tensor(gt_A).long()
        gt_B = torch.tensor(gt_B).long()

        out = {}
        out['img_A1'] = img_A1
        out['img_A2'] = img_A2
        out['img_B1'] = img_B1

        out['label_A1'] = gt_A
        out['label_A2'] = gt_A
        out['label_B1'] = gt_B

        return out
