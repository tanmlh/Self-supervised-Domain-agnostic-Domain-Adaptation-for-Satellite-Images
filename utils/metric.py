import operator
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
import scipy.stats as stats
import torch.nn.functional as F
import torch
import pdb
import pickle

def iou_per_class(outputs: torch.Tensor, labels: torch.Tensor):

    eps = 1e-8
    outputs = outputs.squeeze(1)
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + eps) / (union + eps)

    return iou

def cal_iou(pred_map, label_map, num_classes, cls_idx=None):
    B, H, W = pred_map.shape
    ious = []
    for cls in range(num_classes):
        if cls_idx is None or cls == cls_idx:
            cur_iou = iou_per_class(pred_map == cls, label_map == cls)
            ious.append(cur_iou)

    return torch.stack(ious, dim=1)
