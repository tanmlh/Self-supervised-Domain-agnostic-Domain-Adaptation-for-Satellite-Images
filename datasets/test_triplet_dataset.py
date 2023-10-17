from __future__ import absolute_import

import os
import io
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.transforms import ImageNetPolicy
import pdb
from PIL import Image, ImageFilter, ImageOps
import pickle
import time

from .base_dataset import BaseDataset

class TestTripletDataset(BaseDataset):

    def __init__(self, loader_conf, phase):

        super(TestTripletDataset, self).__init__(loader_conf, phase)

        self.num_classes = loader_conf['num_classes']
        self.num_domains = loader_conf['num_doms']
        self.filter_size = loader_conf['filter_size'] if 'filter_size' in loader_conf else 1
        self.reverse_gt = loader_conf['reverse_gt'] if 'reverse_gt' in loader_conf else False
        self.use_hist_equ = loader_conf['use_hist_equ'] if 'use_hist_equ' in loader_conf else False

        self.dom_imgs = {}

        # augmentation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = []

        transform.append(transforms.ToTensor())
        transform.append(normalize)
        self.transform = transforms.Compose(transform)

        self.dom_map = {}
        dom_cnt = 0
        with open(self.file_list_path) as f:
            lines = f.readlines()
            self.lines = lines
            for line in lines:
                img_path, gt_path, dom_name, is_train = line.strip().split(' ')
                if not dom_name in self.dom_imgs:
                    self.dom_map[dom_name] = dom_cnt
                    dom_cnt += 1
                    self.dom_imgs[dom_name] = []
                self.dom_imgs[dom_name].append((img_path, gt_path))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):

        A_path_img, A_path_gt, dom_A, is_train = self.lines[idx].strip().split(' ')
        img_A = self.read_img(os.path.join(self.root, A_path_img))
        if self.use_hist_equ:
            img_A = ImageOps.equalize(img_A)

        img_A = self.transform(img_A)

        # if A_path_gt != 'None':
        gt_A = self.read_img(os.path.join(self.root, A_path_gt), 'nearest') if int(is_train) else np.zeros(img_A.shape[:2]).astype(np.uint8)
        gt_A = torch.tensor(np.array(gt_A)).long().squeeze()
        if self.reverse_gt:
            gt_A = (gt_A == 0).long()

        out = {}
        out['img_A1'] = img_A
        out['label_map_A1'] = np.array(gt_A)
        out['dom_A1'] = self.dom_map[dom_A]
        out['A_paths'] = [A_path_img, A_path_gt, dom_A, int(is_train)]

        return out

    def save_hist(self):
        for i in self.dom_imgs.keys():
            for img_path, gt_path in self.dom_imgs[i]:
                img = self.read_img(os.path.join(self.root, img_path))
                cur_hist = np.histogram(np.array(img))
