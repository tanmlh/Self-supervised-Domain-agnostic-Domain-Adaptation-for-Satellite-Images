import os
import pickle
import numpy as np
import cv2
from skimage.exposure import match_histograms
import pdb
import tqdm

class DataNormalizer:

    def __init__(self, conf):
        self.norm_type = conf['norm_type']
        self.num_channels = conf['num_channels']
        self.hist_path = conf['hist_path'] if 'hist_path' in conf else None

    def normalize(self, img):

        assert len(img.shape) == 3 and type(img) == np.ndarray
        H, W, C = img.shape

        if self.norm_type == 'gray_equ_v2':

            mean_A = (img_A1.mean() + img_A2.mean()) / 2.0
            img_A1 = (img_A1 * 0.5 / mean_A)
            img_A2 = (img_A2 * 0.5 / mean_A)

            if self.sr_dom is None or dom_B in self.sr_dom:

                mean_B = img_B1.mean()
                img_B1 = (img_B1 * 0.5 / mean_B)

        elif self.norm_type == 'channel_wise':

            chn_maxv = img.max(axis=0).max(axis=0).reshape(1, 1, C)
            chn_minv = img.min(axis=0).min(axis=0).reshape(1, 1, C)
            img = (img - chn_minv) / (chn_maxv - chn_minv)

        elif self.norm_type == 'plain':

            img = img / 255.0

        return img

    def fit_hist(self, data_dict):

        if self.hist_path is not None and os.path.exists(self.hist_path):
            self.hist_map = pickle.load(open(self.hist_path, 'rb'))

        else:

            self.hist_map = {}
            for key, dataset in data_dict.items():
                sum_hist = np.zeros((self.num_channels, 256))
                print('Fitting histogram for data {}...'.format(key))
                for data in tqdm.tqdm(dataset):
                    img = self.normalize(data)
                    img = (img * 255).astype(np.uint8)
                    for chn in range(self.num_channels):
                        cur_hist = np.bincount(img[:, :, chn].flatten(), minlength=256)
                        sum_hist[chn] += cur_hist

                sum_hist = sum_hist / sum_hist.sum(axis=1).reshape((self.num_channels, 1))
                cum_hist = np.cumsum(sum_hist, axis=1)
                self.hist_map[key] = np.floor(255 * cum_hist).astype(np.uint8)

            if self.hist_path is not None and not os.path.exists(self.hist_path):
                pickle.dump(self.hist_map, open(self.hist_path, 'wb'))

    def hist_equalize(self, data, dom):
        img = (data * 255).astype(np.uint8)
        for chn in range(self.num_channels):
            img[:, :, chn] = self.hist_map[dom][chn][img[:, :, chn]]

        return img / 255.


    def hist_match(self, img_A, img_B):
        img_A = match_histograms(img_A, img_B, multichannel=True)
        return img_A

