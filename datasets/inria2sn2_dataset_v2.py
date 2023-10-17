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

class INRIA2SN2Dataset(Dataset):

    def __init__(self, loader_conf, phase='train'):

        super(INRIA2SN2Dataset, self).__init__()

        self.root = loader_conf['dataset_dir']
        self.img_H = loader_conf['img_H']
        self.img_W = loader_conf['img_W']
        self.phase = loader_conf['phase'] if 'phase' in loader_conf else phase
        self.use_resize = loader_conf['use_resize'] if 'use_resize' in loader_conf else True
        self.num_used_data = loader_conf['num_used_data'] if 'num_used_data' in loader_conf else 1e8
        self.aug_conf = loader_conf['aug_conf']
        self.use_hist_equ = loader_conf['use_hist_equ'] if 'use_hist_equ' in loader_conf else False
        self.use_hist_mat = loader_conf['use_hist_mat'] if 'use_hist_mat' in loader_conf else False
        self.hist_mat_prob = loader_conf['hist_mat_prob'] if 'hist_mat_prob' in loader_conf else 0
        self.use_rgb = loader_conf['use_rgb'] if 'use_rgb' in loader_conf else False
        self.norm_type = loader_conf['norm_type'] if 'norm_type' in loader_conf else None
        self.A_sample_type = loader_conf['A_sample_type']
        self.B_sample_type = loader_conf['B_sample_type']
        self.random_type = loader_conf['random_type']
        self.test_reader_conf = loader_conf['test_reader_conf'] if 'test_reader_conf' in loader_conf else None
        self.train_reader_conf = loader_conf['train_reader_conf'] if 'train_reader_conf' in loader_conf else None
        self.dom_map = loader_conf['dom_map'] if 'dom_map' in loader_conf else None

        self.normalizer = data_normalizer.DataNormalizer(loader_conf)
        self.augmentor = data_augmentor.DataAugmentor(self.aug_conf)
        self.train_reader = data_reader.DataReader(self.train_reader_conf) if self.train_reader_conf is not None else None
        self.test_reader = data_reader.DataReader(self.test_reader_conf) if self.test_reader_conf is not None else None

        if self.use_hist_equ:
            iter_train = self.train_reader.get_iterators() if self.train_reader is not None else {}
            iter_test = self.test_reader.get_iterator() if self.test_reader is not None else {}
            self.normalizer.fit_hist({**iter_train, **iter_test})

        transform = []
        transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_reader)
        if self.phase == 'test':
            return len(self.test_reader)
        else:
            return len(self.train_reader) + len(self.test_reader)

    def get_sample(self, sample_type, idx=None):

        if sample_type == 'random':
            flag = random.random() < 0.5
        elif sample_type == 'data1':
            flag = 0
        elif sample_type == 'data2':
            flag = 1
        elif sample_type == 'data12':
            if idx < len(self.train_reader):
                flag = 0
            else:
                flag = 1
                idx -= len(self.train_reader)
        else:
            raise ValueError

        reader = self.train_reader if flag == 0 else self.test_reader

        img, gt, dom = reader.sample(idx)

        if flag == 0:
            img = img[:, :, [2,1,0]]

        return img, gt, dom

    def __getitem__(self, idx):

        if self.phase == 'train' or self.phase == 'gen':
            img_A, gt_A, dom_A = self.get_sample(self.A_sample_type, idx)
            img_B, gt_B, dom_B = self.get_sample(self.B_sample_type, idx)
        else:
            img_A, gt_A, dom_A = self.get_sample(self.A_sample_type, idx)
            img_B, gt_B, dom_B = self.get_sample(self.B_sample_type, idx)

        gt_A = (gt_A >= 128).astype(np.uint8)
        gt_B = (gt_B >= 128).astype(np.uint8)

        img_A  = self.normalizer.normalize(img_A)
        img_B  = self.normalizer.normalize(img_B)

        if self.use_hist_equ:
            img_A = self.normalizer.hist_equalize(img_A, dom_A)
            img_B = self.normalizer.hist_equalize(img_B, dom_B)

        elif self.use_hist_mat:
            if random.random() < self.hist_mat_prob:
                img_A = self.normalizer.hist_match(img_A, img_B)

        img_A1, gt_A1 = self.augmentor.data_aug(img_A, gt_A)
        img_A2, gt_A2 = self.augmentor.data_aug(img_A, gt_A)
        img_B1, gt_B1 = self.augmentor.data_aug(img_B, gt_B)
        img_A1, img_A2 = self.augmentor.color_shift(img_A1, img_A2)
        img_B1 = self.augmentor.color_shift(img_B1)

        img_A1 = self.transform(img_A1.astype(np.float32))
        img_A2 = self.transform(img_A2.astype(np.float32))
        img_B1 = self.transform(img_B1.astype(np.float32))

        gt_A1 = torch.tensor(gt_A1).long()
        gt_A2 = torch.tensor(gt_A2).long()
        gt_B1 = torch.tensor(gt_B1).long()

        out = {}
        out['img_A1'] = img_A1
        out['img_A2'] = img_A2
        out['img_B1'] = img_B1

        out['label_map_A1'] = gt_A1
        out['label_map_A2'] = gt_A2
        out['label_map_B1'] = gt_B1

        if self.dom_map is not None:
            out['dom_A'] = self.dom_map[dom_A]
            out['dom_B'] = self.dom_map[dom_B]

        return out
