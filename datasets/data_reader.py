import h5py
import tqdm
import pdb
import cv2
import numpy as np
import os
import random

class DataReader:
    def __init__(self, conf):

        self.file_path = conf['file_path']
        self.dataset_type = conf['dataset_type']
        self.root_dir = conf['root_dir']
        self.num_used_data = conf['num_used_data'] if 'num_used_data' in conf else 1e9
        self.sample_type = conf['sample_type'] if 'sample_type' in conf else 'linear'

        self.dataset = self.parse_dataset()

    def parse_dataset(self):

        if self.file_path.endswith('.h5'):

            f = h5py.File(self.file_path, 'r')
            self.data = f['sen2']
            self.labels = f['label']
            self.data_len = len(self.data)

        elif self.file_path.endswith('.txt'):
            if self.dataset_type == 'mul_dom':
                dataset = {}

            with open(self.file_path) as f:
                lines = f.readlines()
                self.lines = lines
                tq = tqdm.tqdm(lines)
                tq.set_description('Loading data from {}'.format(self.file_path))

                for i, line in enumerate(tq):

                    if i >= self.num_used_data:
                        break

                    if self.dataset_type == 'mul_dom':

                        img_path, gt_path, dom_name = line.strip().split(' ')

                        if not dom_name in dataset:
                            dataset[dom_name] = []

                        dataset[dom_name].append((img_path, gt_path))

        return dataset

    def __len__(self):
        if self.dataset_type == 'mul_dom':
            lens = sum([len(val) for key, val in self.dataset.items()])

        return lens


    def get_dom_idx(self, dataset, idx):

        lens = [len(dataset[x]) for x in dataset.keys()]
        sum_len = 0
        for i, x in enumerate(lens):
            if idx < sum_len + x:
                return list(dataset.keys())[i], idx - sum_len
            sum_len += x

        raise ValueError('Invalid index')

    def sample(self, idx=None):

        if self.sample_type == 'dom':
            dom = random.choice(list(self.dataset.keys()))
            idx = random.choice(range(len(self.dataset[dom])))

        elif self.sample_type == 'uniform':
            weights = [len(x) for key, x in self.dataset.items()]
            dom = random.choices(list(self.dataset.keys()), k=1, weights=weights)[0]
            idx = random.choice(range(len(self.dataset[city])))

        elif self.sample_type == 'linear':
            dom, idx = self.get_dom_idx(self.dataset, idx)

        else:
            raise ValueError('No such sample type!')


        img_path, gt_path = self.dataset[dom][idx]

        img = self.read_data(os.path.join(self.root_dir, img_path))
        gt = self.read_data(os.path.join(self.root_dir, gt_path))

        return img, gt, dom

    def read_data(self, data_path):

        data = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)

        return data

    def get_iterator(self):

        if self.dataset_type == 'mul_dom':
            iterators = {}
            for dom, data in self.dataset.items():
                iterators[dom] = [self.sample(idx)[0] for idx in range(len(data))]

            return iterators

