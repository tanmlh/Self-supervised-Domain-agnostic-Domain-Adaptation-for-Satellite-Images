import cv2
import numpy as np
import torch
import random
import pdb

class DataAugmentor:

    def __init__(self, aug_conf):
        self.aug_conf = aug_conf

    def data_aug(self, img, mask=None):
        '''
        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        use_mask = mask is not None
        crop_w, crop_h = self.aug_conf['crop_size']

        if self.aug_conf['random_rotate']:
            times = random.choice([0, 1, 2, 3])
            img = np.rot90(img, k=times)
            if use_mask: mask = np.rot90(mask, k=times)

        if self.aug_conf['random_mirror']:
            # random mirror
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                if use_mask: mask = cv2.flip(mask, 1)

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
            if use_mask: mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

            # pad crop
            if ow < crop_w or oh < crop_h:
                padh = crop_h - oh if oh < crop_h else 0
                padw = crop_w - ow if ow < crop_w else 0

                img = cv2.copyMakeBorder(img, padh//2, padh - padh//2, padw//2, padw-padw//2,  cv2.BORDER_REFLECT_101)
                if use_mask: mask = cv2.copyMakeBorder(mask, padh//2, padh - padh//2, padw//2, padw-padw//2,  cv2.BORDER_REFLECT_101)

            # random crop crop_size
            h, w = img.shape[:2]
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            img = img[x1:x1+crop_w, y1:y1+crop_h]
            if use_mask: mask = mask[x1:x1+crop_w, y1:y1+crop_h]

        elif self.aug_conf['resize']:
            img = cv2.resize(img, self.aug_conf['crop_size'], cv2.INTER_LINEAR)
            if use_mask: mask = cv2.resize(mask, self.aug_conf['crop_size'], interpolation=cv2.INTER_NEAREST)

        if self.aug_conf['gaussian_blur']:
            # gaussian blur as in PSP
            if random.random() < 0.5:
                # img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
                k_size = random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (k_size, k_size), cv2.BORDER_DEFAULT)

        return img, mask

    def hsv_shift(self, img, h_shift, s_shift, v_shift):

        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if h_shift != 0:
            lut_hue = np.arange(0, 256, dtype=np.int16)
            lut_hue = np.mod(lut_hue + h_shift, 180).astype(np.uint8)
            hue = cv2.LUT(hue, lut_hue)

        if s_shift != 0:
            lut_sat = np.arange(0, 256, dtype=np.int16)
            lut_sat = np.clip(lut_sat + s_shift, 0, 255).astype(np.uint8)
            sat = cv2.LUT(sat, lut_sat)

        if v_shift != 0:
            lut_val = np.arange(0, 256, dtype=np.int16)
            lut_val = np.clip(lut_val + v_shift, 0, 255).astype(np.uint8)
            val = cv2.LUT(val, lut_val)

        img = cv2.merge((hue, sat, val)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB) / 255.

        return img


    def color_shift(self, img1, img2=None):

        if  'hsv_shift' in self.aug_conf:
            h_range, s_range, v_range, prob = self.aug_conf['hsv_shift']

            h_shift = random.randint(-h_range, h_range)
            s_shift = random.randint(-s_range, s_range)
            v_shift = random.randint(-v_range, v_range)

            if random.random() < prob:
                img1 = self.hsv_shift(img1, h_shift, s_shift, v_shift)
                if img2 is not None: img2 = self.hsv_shift(img2, h_shift, s_shift, v_shift)

        if 'gamma_shift' in self.aug_conf:
            low, high, prob = self.aug_conf['gamma_shift']
            gamma = random.randint(low, high) / 100

            img1 = np.power(img1, gamma)
            if img2 is not None: img2 = np.power(img2, gamma)

        if img2 is not None:
            return img1, img2
        else:
            return img1
