import cv2
import io
import os
import os.path as osp
from PIL import Image
import numpy as np
import random
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import math
import pdb
import torch

def resize4gt_map(gt_map, num_classes):
    B, H, W = gt_map.shape
    one_hot_map = torch.zeros(B, num_classes, H, W)
    one_hot_map.scatter_()

class RandomDoubleFlip:
    def __init__(self):
        pass

    def __call__(self, img_A, img_B):

        img_A2 = img_A
        img_B2 = img_B
        if random.random() < 0.5:
            img_A2 = TF.hflip(img_A2)
            img_B2 = TF.hflip(img_B2)

        if random.random() < 0.5:
            img_A2 = TF.vflip(img_A2)
            img_B2 = TF.vflip(img_B2)

        return img_A2, img_B2

class RandomDoubleRotate:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img_A, img_B):
        angle = random.choice(self.angles)

        img_A2 = TF.rotate(img_A, angle)
        img_B2 = TF.rotate(img_B, angle)

        return img_A2, img_B2

def cv2_resize_no_pad(img, height, max_width, phase, random_scale=True):
    h, w = img.shape[:2]
    new_w = height / h * w
    if phase=='train' and random_scale:
        new_w = new_w * random.uniform(0.5, 2.0)
    new_w = min(round(new_w)+1, max_width)
    img = cv2.resize(img, (new_w, height))
    return img, [1.0 * height / h, 1.0 * new_w / w]

def cv2_resize_scale(img, H, W):
    img_H, img_W, _ = img.shape
    img = cv2.resize(img, (W, H))
    ratio = [1.0 * H / img_H, 1.0 * W / img_W]

    return img, ratio

def cv2_resize_scale_for_test(img, H, W):
    img_H, img_W, _ = img.shape
    if img_W / img_H <= 4:
        img = cv2.resize(img, (W, H))
        ratio = [H / img_H, W / img_W]
    else:
        temp = H * img_W / img_H
        temp = round(max(temp // 32, 1) * 32)
        img = cv2.resize(img, (temp, H))

        # img = np.pad(img, ((0, 0), (0, new_W - temp), (0, 0)), 'constant', constant_values = 0)

        ratio = [H / img_H, temp / img_W]

    return img, ratio

def cv2_resize_pad(img, H, W):
    # pdb.set_trace()
    img_H, img_W, _ = img.shape
    if img_H / img_W > H / W:
        ratio = H / img_H
    else:
        ratio = W / img_W

    img_res = cv2.resize(img, (round(img_W * ratio), round(img_H * ratio)))

    new_H, new_W, _ = img_res.shape

    pos_trans = [ratio]

    if img_H / img_W > H / W:
        img_res = np.pad(img_res, ((0, 0), ((W - new_W) // 2, (W - new_W) - (W - new_W) // 2), (0, 0)),
                         'constant', constant_values = 0)

        offset = np.array([[(W - new_W) // 2], [0]])
        pos_trans.append(np.tile(offset, [1, 4]))

    else:
        img_res = np.pad(img_res, (((H - new_H) // 2, (H - new_H) - (H - new_H) // 2), (0, 0), (0, 0)),
                         'constant', constant_values = 0)

        offset = np.array([[0], [(H - new_H) // 2]])
        pos_trans.append(np.tile(offset, [1, 4]))

    return img_res, pos_trans

## preprocess
def preprocess(img, phase, color, expand_ratio=0.05):
    img = np.array(img)
    if phase != 'train' or not color:
        return img
    else:
        if random.random() < 0.5:
            img = random_reverse(img)
        if random.random() < 0.5:
            img = random_expand(img, expand_ratio)
        return img

def random_expand(img, expand_ratio):
    h, w, _ = img.shape
    expand_w = expand_ratio * w
    ratios = list(np.linspace(0.1, 1.0, 10))
    choices = [random.choice(ratios) for i in range(10)]
    img = np.pad(img, ((int(expand_w*choices[0]), int(expand_w*choices[1])), (int(expand_w*choices[2]), int(expand_w*choices[3])), (0, 0)), \
             'constant', constant_values=255)
    return img

def random_reverse(img):
    return 255 - img

## text augmentation
class TextPolicy(object):
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.range = {
            "color": np.linspace(0.6, 1.5, 10),
            "contrast": np.linspace(0.6, 1.5, 10),
            "sharpness": np.linspace(0.1, 1.9, 10),
            "brightness": np.linspace(0.5, 1.4, 10),
            "autocontrast": [0] * 10,
            "gauss_blur": np.linspace(0.5, 1.0, 10),
            "rotate": np.linspace(-20, 20, 10),
            # "motion_blur": [1, 1, 1 ,1, 2, 2, 3, 3, 4, 5],
            # "perspective": [0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
        }

    def func(self, op, img, magnitude):
        if op=="color": return ImageEnhance.Color(img).enhance(magnitude)
        elif op=="contrast": return ImageEnhance.Contrast(img).enhance(magnitude)
        elif op=="sharpness": return ImageEnhance.Sharpness(img).enhance(magnitude)
        elif op=="brightness": return ImageEnhance.Brightness(img).enhance(magnitude)
        elif op=="autocontrast": return ImageOps.autocontrast(img)
        elif op== "rotate": return self.rotate(img, magnitude)
        elif op=="gauss_blur": return img.filter(ImageFilter.GaussianBlur(radius=magnitude))
        # elif op=="motion_blur": return self.motion_blur(img, magnitude)
        # elif op=="perspective": return self.perspective(img, magnitude)
        else:
            print('error ops')

    def __call__(self, img):
        if random.random() < self.keep_prob:
            return img, None
        else:
            rand = np.random.randint(0, 10, 2)
            policies = random.sample(list(self.range.keys()), 2)
            rot_param = None

            res = self.func(policies[0], img, self.range[policies[0]][rand[0]])
            if type(res) is not tuple:
                img = res
            else:
                img, rot_param = res

            if random.random() < 0.5:
                res = self.func(policies[1], img, self.range[policies[1]][rand[1]])

            if type(res) is not tuple:
                img = res
            else:
                img, rot_param = res

            return img, rot_param

    def rotate(self, img, magnitude):
        # w, h = img.size
        # img = img.rotate(magnitude, expand=True)
        # img = img.resize((w, h), Image.ANTIALIAS)
        img = np.array(img)
        H, W, C = img.shape
        center_H, center_W = H // 2, W // 2
        M = cv2.getRotationMatrix2D((center_W, center_H), magnitude, 1)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_W = int((H * sin) + (W * cos))
        new_H = int((H * cos) + (W * sin))
        M[0, 2] += (new_W / 2) - center_W
        M[1, 2] += (new_H / 2) - center_H

        img = cv2.warpAffine(img, M, (new_W, new_H))
        img = Image.fromarray(img)
        return img, M

    '''
    def motion_blur(self, image, degree=6, angle=45):
        image = np.array(image)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        print(blurred.shape)
        blurred = torchvision.transforms.ToPILImage()(blurred)
        return blurred
    '''

    def perspective(self, image, select=0, udeg=10, ddeg=10, ldeg=20, rdeg=20, starPoints = 0):
        img = np.array(image)

        LEN_QUANTITY = 30
        height, width = img.shape[0], img.shape[1]
        if starPoints == 0:
            starPoints = []
            starPoints.append((0,0))
            starPoints.append((width-1,0))
            starPoints.append((0,height-1))
            starPoints.append((width-1,height-1))

        selectPoints = []
        selectedLU = (starPoints[0][0],starPoints[0][1])
        selectedRU = (starPoints[1][0],starPoints[1][1])
        selectedLD = (starPoints[2][0],starPoints[2][1])
        selectedRD = (starPoints[3][0],starPoints[3][1])

        selectPoints.append(selectedLU)
        selectPoints.append(selectedRU)
        selectPoints.append(selectedLD)
        selectPoints.append(selectedRD)

        if select == 0:
        	sinn = math.sin(udeg*math.pi/180)
        	coss = math.cos(udeg*math.pi/180)
        	showWidth = int(height*sinn/LEN_QUANTITY*width)
        	showHeight = int(height*coss)
        	showConors = []
        	showLU = (int(showWidth*0.15), 0)
        	showRU = (int(width-showWidth*0.15), 0)
        	showLD = (0, showHeight-1)
        	showRD = (width-1, showHeight-1)
        	showWidth = width
        if select == 1:
        	sinn = math.sin(ddeg*math.pi/180)
        	coss = math.cos(ddeg*math.pi/180)
        	showWidth = int(height*sinn/LEN_QUANTITY*width)
        	showHeight =int(height*coss)
        	showConors = []
        	showLU = (0,0)
        	showRU = (width-1, 0)
        	showLD = (int(showWidth*0.15),showHeight-1)
        	showRD = (int(-showWidth*0.15+width), showHeight-1)
        	showWidth = width
        if select == 2:
        	sinn = math.sin(ldeg*math.pi/180)
        	coss = math.cos(ldeg*math.pi/180)
        	showWidth = int(width*coss)
        	showHeight= int(width*sinn/LEN_QUANTITY*height)
        	showLU = (0, 0)
        	showRU = (showWidth-1, int(showHeight*0.15))
        	showLD = (0,height-1)
        	showRD = (showWidth-1,  int(height-showHeight*0.15))
        	showHeight = height
        if select == 3:
        	sinn = math.sin(rdeg*math.pi/180)
        	coss = math.cos(rdeg*math.pi/180)
        	showWidth = int(width*coss)
        	showHeight=int(width*sinn/LEN_QUANTITY*height)
        	showLU = (0, int(showHeight*0.15))
        	showRU = (showWidth-1, 0)
        	showLD = (0, int(height - showHeight*0.15))
        	showRD = (showWidth-1, height-1)
        	showHeight = height
        showConors = []

        showConors.append(showLU)
        showConors.append(showRU)
        showConors.append(showLD)
        showConors.append(showRD)

        transform = cv2.getPerspectiveTransform(np.array(selectPoints, dtype=np.float32), np.array(showConors, dtype=np.float32))
        img = cv2.warpPerspective(img, transform,(showWidth,showHeight))
        img = cv2.resize(img, (width, height))
        img = Image.fromarray(img)
        return img

class ImageNetPolicy(object):
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.range = {
            "shearX": np.linspace(-0.1, 0.1, 10),
            "shearY": np.linspace(-0.1, 0.1, 10),
            "rotate": np.linspace(-10, 10, 10),
            "color": np.linspace(0.6, 1.5, 10),
            "posterize": np.round(np.linspace(4, 8, 10), 0).astype(np.int),
            "contrast": np.linspace(0.6, 1.5, 10),
            "sharpness": np.linspace(0.1, 1.9, 10),
            "brightness": np.linspace(0.5, 1.4, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "blur": np.linspace(0.5, 1.0, 10),
            "detail": [0] * 10
        }

    def func(self, op, img, magnitude):
        if op== "shearX": return shear(img, magnitude * 180, direction="x")
        elif op== "shearY": return shear(img, magnitude * 180, direction="y")
        elif op== "rotate": return img.rotate(magnitude)
        elif op=="color": return ImageEnhance.Color(img).enhance(magnitude)
        elif op=="posterize": return ImageOps.posterize(img, magnitude)
        elif op=="contrast": return ImageEnhance.Contrast(img).enhance(magnitude)
        elif op=="sharpness": return ImageEnhance.Sharpness(img).enhance(magnitude)
        elif op=="brightness": return ImageEnhance.Brightness(img).enhance(magnitude)
        elif op=="autocontrast": return ImageOps.autocontrast(img)
        elif op=="equalize": return ImageOps.equalize(img)
        elif op=="blur": return img.filter(ImageFilter.GaussianBlur(radius=magnitude))
        elif op=="detail": return img.filter(ImageFilter.DETAIL)
        else: print('error ops')

    def __call__(self, img):
        if random.random() < self.keep_prob:
            return img
        else:
            rand = np.random.randint(0, 10, 2)
            policies = random.sample(list(self.range.keys()), 2)
            img = self.func(policies[0], img, self.range[policies[0]][rand[0]])
            if random.random() < 0.5:
                img = self.func(policies[1], img, self.range[policies[1]][rand[1]])
            return img


def shear(img, angle_to_shear, direction="x"):
    width, height = img.size
    phi = math.tan(math.radians(angle_to_shear))

    if direction=="x":
        shift_in_pixels = phi * height

        if shift_in_pixels > 0:
            shift_in_pixels = math.ceil(shift_in_pixels)
        else:
            shift_in_pixels = math.floor(shift_in_pixels)

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)

        img = img.transform((int(round(width + shift_in_pixels)), height),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC, fillcolor=(128, 128, 128))

        return img.resize((width, height), resample=Image.BICUBIC)

    elif direction == "y":
        shift_in_pixels = phi * width

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0, phi, 1, -matrix_offset)

        image = img.transform((width, int(round(height + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC, fillcolor=(128, 128, 128))

        return image.resize((width, height), resample=Image.BICUBIC)


if __name__=='__main__':
    img = Image.open('1.jpg').convert('L')
    policy = TextPolicy(keep_prob=0.0)
    img = policy(img)
    print(np.array(img).shape)
