import os
import time
import numpy as np
import torch
from PIL import Image
import pdb
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

name_classes = [
    'background',
    'road'
]

class Evaluator():
    def __init__(self, dataset_name='planet', log_dir=None):
        self.num_class = 2 if dataset_name == 'planet' else 2

        self.dataset_name = dataset_name
        self.log_dir = log_dir

        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.ignore_index = None
        self.num_bins = 10000
        self.hist = np.zeros((self.num_class, self.num_bins))


    def Pixel_Accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return PA

    def Mean_Pixel_Accuracy(self):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        MPA = np.nanmean(MPA[:self.ignore_index])

        return MPA

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        MIoU = np.nanmean(MIoU[:self.ignore_index])

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        FWIoU = np.multiply(np.sum(self.confusion_matrix, axis=1), np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                         np.diag(self.confusion_matrix))

        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)

        return FWIoU

    def Mean_Precision(self):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision

    def Print_Every_class_Eval(self, name, logging=None):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        F1_score = 2 * Precision * Recall / (Precision + Recall)
        Class_ratio = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        logging.info('===>{:<12}:\t'.format('Everyclass') + 'MPA\t' + 'MIoU\t' + 'PC\t' + 'Ratio\t'
                     + 'Pred_R\t' + 'F1\t' + 'Recall')

        names = name_classes

        for ind_class in range(len(MIoU)):
            pa = str(round(MPA[ind_class] * 100, 2)) if not np.isnan(MPA[ind_class]) else 'nan'
            iou = str(round(MIoU[ind_class] * 100, 2)) if not np.isnan(MIoU[ind_class]) else 'nan'
            pc = str(round(Precision[ind_class] * 100, 2)) if not np.isnan(Precision[ind_class]) else 'nan'
            cr = str(round(Class_ratio[ind_class] * 100, 2)) if not np.isnan(Class_ratio[ind_class]) else 'nan'
            pr = str(round(Pred_retio[ind_class] * 100, 2)) if not np.isnan(Pred_retio[ind_class]) else 'nan'
            f1 = str(round(F1_score[ind_class] * 100, 2)) if not np.isnan(F1_score[ind_class]) else 'nan'
            rc = str(round(Recall[ind_class] * 100, 2)) if not np.isnan(Recall[ind_class]) else 'nan'

            logging.info('===>{:<12}:\t'.format(names[ind_class]) + pa + '\t' + iou + '\t' + pc +
                         '\t' + cr + '\t' + pr + '\t' + f1 + '\t' + rc + '\t')
    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def add_hist(self, prob_map):
        B, C, H, W = prob_map.shape
        assert C == self.num_class

        prob_map = torch.tensor(prob_map).numpy()
        label = prob_map.argmax(axis=1)
        for i in range(C):
            self.hist[i] += np.histogram(prob_map[:, i][label==i], bins=self.num_bins, range=(0, 1))[0]

    def add_hist2(self, dis_map, pred_map):
        B, C, H, W = pred_map.shape
        pred_label = pred_map.max(dim=1)[1]
        for i in range(C):
            temp = dis_map[pred_label==i]
            self.hist[i] += np.histogram(temp.numpy(), bins=self.num_bins, range=(0, 1))[0]

    def print_hist(self):
        int_hist = np.zeros((self.num_class, self.num_bins))
        bins = np.linspace(0, 1, self.num_bins+1)

        gamma = 0.3
        thres = []
        for i in range(self.num_class):
            for j in range(self.num_bins):
                if j > 0:
                    int_hist[i, j] = int_hist[i, j-1]
                int_hist[i, j] += self.hist[i, j]

            int_hist[i] /= sum(self.hist[i])
            for j in range(self.num_bins):
                if int_hist[i, j] > (1-gamma):
                    thres.append(bins[j])
                    break
        print(thres)
        with open('thres.txt', 'a') as f:
            f.write(self.dataset_name + ' gamma=' + str(gamma) + ':\n')
            for thre in thres:
                f.write(str(thre) + ', ')
            f.write('\n\n')
            f.close()

    def add_sim_hist(self, sim_dist):
        num_bins = 25
        if not hasattr(self, 'sim_hist'):
            self.sim_hist = np.zeros((4, num_bins))
        for i in range(4):
            self.sim_hist[i] += np.histogram(sim_dist[i].cpu().numpy(), bins = num_bins, range=(0, 1))[0]

    def add_loc_hist(self, loc_rank):
        if not hasattr(self, 'loc_hist'):
            self.loc_hist = np.zeros((len(loc_rank), 2))

        for i, rank in enumerate(loc_rank):
            self.loc_hist[i][0] += rank[0]
            self.loc_hist[i][1] += rank[1]

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.hist = np.zeros((self.num_class, self.num_bins))

    def val_info(self, name, logger):

        PA = self.Pixel_Accuracy()
        MPA = self.Mean_Pixel_Accuracy()
        MIoU = self.Mean_Intersection_over_Union()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()
        PC = self.Mean_Precision()
        logger.info("########## Eval on {} ############".format(name))
        logger.info('PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.4f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(PA, MPA, MIoU, FWIoU, PC))

        return PA, MPA, MIoU, FWIoU


def softmax(k, axis=None):
    exp_k = np.exp(k)
    return exp_k / np.sum(exp_k, axis=axis, keepdims=True)

