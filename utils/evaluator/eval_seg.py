import logging
import pdb
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import numpy as np
import h5py
import scipy.io as scio


class Evaluator:

    def __init__(self, conf):

        self.num_classes = conf['num_classes']
        self.dataset_name = conf['dataset_name']
        self.conf_mat = np.zeros((self.num_classes, self.num_classes))
        self.class_names = conf['class_names']

        if not os.path.exists(conf['log_dir']):
            os.makedirs(conf['log_dir'])
        log_file_path = os.path.join(conf['log_dir'], 'log.txt')
        logging.basicConfig(filename=log_file_path,
                            filemode='a', level=logging.INFO)

    def add_batch(self, preds, gts):

        assert preds.shape == gts.shape

        cur_mat = confusion_matrix(preds, gts, labels=np.arange(self.num_classes))
        self.conf_mat += cur_mat

    def reset(self):
        self.conf_mat = np.zeros((self.num_classes, self.num_classes))


    def evaluate(self):

        OA = np.sum(np.diag(self.conf_mat)) / np.sum(self.conf_mat)
        IoU = self.cal_IoU()

        return {'scalar|OA': OA, 'scalar|mIoU': np.nanmean(IoU), 'scalar|building_IoU': IoU[1]}


    def cal_kappa(self):
        pe_rows = np.sum(self.conf_mat, axis=0)
        pe_cols = np.sum(self.conf_mat, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(self.conf_mat) / float(sum_total)

        return (po - pe) / (1 - pe)

    def cal_IoU(self):
        IoU = np.diag(self.conf_mat) / \
              (np.sum(self.conf_mat, axis=1)
               + np.sum(self.conf_mat, axis=0)
               - np.diag(self.conf_mat))

        return IoU

    def cal_building_IoU(self):
        return cal_IoU()[1]

    def print_info(self):

        OA = np.sum(np.diag(self.conf_mat)) / np.sum(self.conf_mat)

        AA = np.diag(self.conf_mat) / self.conf_mat.sum(axis=1)
        AA = np.nanmean(AA)

        Kappa = self.cal_kappa()
        IoU = self.cal_IoU()

        Precision = np.diag(self.conf_mat) / self.conf_mat.sum(axis=0)
        Recall = np.diag(self.conf_mat) / self.conf_mat.sum(axis=1)
        F1_score = 2 * Precision * Recall / (Precision + Recall)

        logging.info("########## Eval on {} ############".format(self.dataset_name))
        logging.info('OA:{:.4f}\t AA:{:.4f}\t Ka:{:.4f}\t F1:{:.4f}\t mIoU:{:.4f}'\
                     .format(OA, AA, Kappa, np.nanmean(F1_score), np.nanmean(IoU)))

        logging.info('{:<25}{:<10}{:<10}{:<10}{:<10}'.format('Everyclass', 'PC', 'RC', 'F1', 'IoU'))
        for ind_class in range(self.num_classes):
            pc = str(round(Precision[ind_class] * 100, 2)) if not np.isnan(Precision[ind_class]) else 'nan'
            rc = str(round(Recall[ind_class] * 100, 2)) if not np.isnan(Recall[ind_class]) else 'nan'
            f1 = str(round(F1_score[ind_class] * 100, 2)) if not np.isnan(F1_score[ind_class]) else 'nan'
            iou = str(round(IoU[ind_class] * 100, 2)) if not np.isnan(IoU[ind_class]) else 'nan'
            logging.info('{:<25}{:<10}{:<10}{:<10}{:<10}'.format(self.class_names[ind_class], pc, rc, f1, iou))
