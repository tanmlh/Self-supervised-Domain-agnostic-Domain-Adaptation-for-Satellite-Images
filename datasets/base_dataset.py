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
from PIL import Image
import pickle
import time
import cv2
from scipy import misc

class BaseDataset(Dataset):
    def __init__(self, loader_conf, phase):
        super(BaseDataset, self).__init__()
        self.root = loader_conf['dataset_dir']
        self.file_list_path = loader_conf['file_list_path']
        self.img_H = loader_conf['img_H']
        self.img_W = loader_conf['img_W']
        self.use_imagenet_aug = loader_conf['use_imagenet_aug'] if 'use_imagenet_aug' in loader_conf else False
        self.use_mc = loader_conf['use_mc'] if 'use_mc' in loader_conf else False
        self.phase = phase
        self.use_augment = loader_conf['use_augment']
        self.use_resize = loader_conf['use_resize'] if 'use_resize' in loader_conf else True

    def read_img(self, file_path, resize_type='default'):
        img = np.array(Image.open(file_path))
        # img = cv2.imread(file_path)
        # if img.shape[-1] == 3:
        #     img = img[:, :, [2, 1, 0]]

        if self.use_resize:
            if resize_type == 'default':
                img = cv2.resize(img, (self.img_W, self.img_H), cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (self.img_W, self.img_H), cv2.INTER_NEAREST)

        return img
