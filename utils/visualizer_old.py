import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pdb
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def visualize(vis_type, vis_data):
    if vis_type == 'vis|proto':
        return vis_proto(*vis_data)
    if vis_type == 'vis|ent_map':
        return visualizer.get_image_loc_map(*vis_data)

# def vis_proto(feat_map, proto, dom_A):
#     B, _, feat_dim = feat_map.shape
#     num_dom, _, _ = proto.shape
#     num_pts = 100
#     cmap = plt.cm.prism(range(num_dom))[:, :3] # (label_len, 3)
# 
#     sample_pixel = random.choices(range(feat_map.shape[1]), k=num_pts)
#     feat_map[:, sample_pixel, :] # (B, 100, feat_dim)
#     fig, ax = plt.figure()
#     ax = Axes3D(fig)
#     for i in range(B):
#         for j in range(num_pts):
#             ax.scatter(feat_map[i, :, 0], feat_map[i, :, 1], feat_map[i, :, 2], '.', alpha=0.1,
#                        c=cmap[dom_A[i]:dom_A[i]+1])
# 
#     for i in range(num_dom):
#         ax.scatter(proto[i, 0], proto[i, 1], proto[i, 2], 's', alpha=0.1,
#                    c=cmap[i:i+1])

def vis_proto(feat_map, label_map, proto, dom_A):
    # feat_map = F.normalize(feat_map, dim=1)
    # proto = F.normalize(proto, dim=1)

    B, _, feat_dim = feat_map.shape
    num_dom, _, _ = proto.shape # (num_dom, num_feat, num_cls)
    proto = proto[:, :, 1]
    dom_A = dom_A + 1
    num_pts = 500
    cmap = plt.cm.Set3(range(num_dom + 1))[:, :3] # (label_len, 3)
    sample_pixel = random.choices(range(feat_map.shape[1]), k=num_pts)
    feat_map = feat_map[:, sample_pixel, :] # (B, num_pts, feat_dim)
    label_map = label_map[:, sample_pixel]

    pca = PCA(n_components=2).fit(proto[[0, 2, 3, 4]])
    p_proto = pca.transform(proto)
    p_feat_map = pca.transform(feat_map.view(-1, feat_dim)).reshape((B, num_pts, 2))

    fig = plt.figure()
    for i in range(B):
        color = cmap[label_map[i] * dom_A[i]]
        plt.scatter(p_feat_map[i, :, 0], p_feat_map[i, :, 1], marker='.', alpha=0.5,
                    c=color)

    for i in range(num_dom):
        plt.scatter(p_proto[i, 0], p_proto[i, 1], marker='s', alpha=1,
                    c=cmap[i+1:i+2])

    return 'figure|vis_proto', fig


def vis_hist(imgs_A, imgs_A2M, label, num_doms, save_path):

    imgs_A = de_normalize(imgs_A)
    imgs_A = imgs_A.clip(0, 1)
    imgs_A2M = de_normalize(imgs_A2M)
    imgs_A2M = imgs_A2M.clip(0, 1)

    dom_hists_A = {}
    dom_hists_A2M = {}
    label = np.array(label)

    for dom in range(num_doms):
        temp_A = imgs_A[label == dom, 2, :, :] * 256
        temp_A2M = imgs_A2M[label == dom, 2, :, :] * 256

        if temp_A.shape != (0,):

            hist_A = np.histogram(temp_A, range=(0, 256), bins=256)[0]
            hist_A2M = np.histogram(temp_A2M, range=(0, 256), bins=256)[0]

            dom_hists_A[dom] = hist_A
            dom_hists_A2M[dom] = hist_A2M

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('original images')
    ax2.set_title('adapted images')

    for i in range(num_doms):
        ax1.plot(dom_hists_A[i])
        ax2.plot(dom_hists_A2M[i])

    fig.savefig(os.path.join(save_path, 'hist.png'))


def vis_tsne(img_As, img_A2Ms, label, num_cls, save_path):
    # embed_A = PCA(n_components=50).fit_transform(feats_A.numpy())
    # embed_A2M = PCA(n_components=50).fit_transform(feats_A2M.numpy())

    feats_A = img_As.view(img_As.shape[0], -1)
    feats_A2M = img_A2Ms.view(img_A2Ms.shape[0], -1)

    embed_A = TSNE(n_components=2).fit_transform(feats_A)
    embed_A2M = TSNE(n_components=2).fit_transform(feats_A2M)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    label = np.array(label)

    ax1.set_title('original images')
    ax2.set_title('adapted images')
    for cls in range(num_cls):
        temp_A = embed_A[label == cls]
        temp_A2M = embed_A2M[label == cls]
        if temp_A.shape != (0,):
            ax1.scatter(temp_A[:, 0], temp_A[:, 1])
            ax2.scatter(temp_A2M[:, 0], temp_A2M[:, 1])

    fig.savefig(os.path.join(save_path, 'tsne.png'))


def np2tensor(np_img):
    H, W, C = np_img.shape
    img = torch.tensor(np_img).permute([2, 0, 1])
    return img

def de_normalize(image):
    B, C, H, W = image.shape

    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1).unsqueeze(0)

    image = image * std + mean

    return image

def merge_list(img_list):
    """
    img_list: list of tensor with shape (C, H, W), where only W varies
    """
    img_W_list = [img.shape[2] for img in img_list]
    img_W = max(img_W_list)
    res_list = []

    for img in img_list:
        if img.shape[2] != img_W:
            pad_img = np.pad(img, ((0, 0), (0, 0), (0, img_W - img.shape[2])), 'constant', constant_values = 0)
            res_list.append(torch.from_numpy(pad_img))
        else:
            res_list.append(img)

    return torch.cat(res_list, dim=1)

def split_batch(image, dim=2):
    """
    Input:
        image: (B, 3, H, W)

    Output:
        tiled_image: (3, B * H, W)
    """
    B, C, H, W = image.shape
    img_list = []
    for i in range(min(B, 16)):
        img_list.append(image[i])
        if i != min(B, 16) - 1:
            if dim == 2:
                img_list.append(torch.zeros(C, H, 5))
            elif dim == 1:
                img_list.append(torch.zeros(C, 5, W))

    return torch.cat(img_list, dim=dim)

def concat(image1, image2):
    """
    Input:
        image1: (3, H, W)
        image2: (3, H, W)
    """
    return torch.cat([image1, image2], dim=2)

def vis_image(img):
    img = de_normalize(img)
    img = img.clip(0, 1)
    return split_batch(img, 1)


def get_image_enc_att(image, enc_att, downsample=1):
    B, num_heads, label_len, h = enc_att.shape
    B, C, H, W = image.shape

    assert h * downsample == H
    assert C == 3

    w = int(W / downsample)

    att = enc_att.mean(dim=1)
    img = de_normalize(image)

    temp = att.view(B, label_len, 1, h, 1).repeat(1, 1, 3, 1, w).contiguous()
    temp = temp.view(B * label_len, 3, h, w)
    temp = F.interpolate(temp, size=(H, W)).view(B, label_len, 3, H, W)
    temp = temp * 0.2 / temp.mean()

    temp_img = img.view(B, 1, 3, H, W).repeat(1, label_len, 1, 1, 1)


    merged_img = (temp  + temp_img).clamp(0, 1)
    img_list = []
    for i in range(min(B, 16)):
        img_list.append(merged_img[i])

    temp = torch.cat(img_list, dim=2) # (label_len, 3, B * H, W)

    img_list = []
    for i in range(min(label_len, 16)):
        img_list.append(temp[i])

    temp = torch.cat(img_list, dim=2)

    return temp

def get_image_pos_range(image, pos_range):
    # point type
    if pos_range.shape[-1] == 2:
        pos_left = pos_range[:, :, 0:1]
        pos_right = pos_range[:, :, 1:]

        # pdb.set_trace()
        cat_left = torch.cat([torch.zeros_like(pos_left), pos_left], dim=2)
        cat_right = torch.cat([torch.zeros_like(pos_right), pos_right], dim=2)

        cat_all = torch.cat([cat_left, cat_right], dim=1)

        return get_image_cen_pos(image, cat_all)

    # circle type
    if pos_range.shape[-1] == 4:
        image = de_normalize(image)
        B, C, H, W = image.shape

        radius = pos_range[:, :, 3]
        cen_pos = pos_range[:, :, :2]

        center_pos_y, center_pos_x = np.where(np.ones((H, W)) > 0) 
        center_pos_y = np.expand_dims(center_pos_y, 0)
        center_pos_x = np.expand_dims(center_pos_x, 0)
        center_pos_y = np.expand_dims(center_pos_y, 2) # (1, H*W, 1)
        center_pos_x = np.expand_dims(center_pos_x, 2)

        bb_center_x = np.expand_dims(cen_pos[:, :, 0], 1) # (B, 1, label_len)
        bb_center_y = np.expand_dims(cen_pos[:, :, 1], 1)

        bb_center_x = (bb_center_x + 1) / 2 * W
        bb_center_y = (bb_center_y + 1) / 2 * H

        ord_dis = (center_pos_x - bb_center_x) ** 2 + (center_pos_y - bb_center_y) ** 2 # (B, H*W, label_len)
        ord_dis = torch.tensor(ord_dis).float()
        ring = torch.abs((ord_dis ** 0.5) - radius.unsqueeze(1)) < 0.3
        ring = ((ord_dis ** 0.5) - radius.unsqueeze(1)) < 0.01
        ring = ring.float().unsqueeze(1)

        vis_image = torch.clamp(image + ring, 0, 1)

        return split_batch(vis_image)


def get_image_cen_pos(image, cen_pos, label=None, sigma=1):
    # cen_pos: (B, label_len, 2)
    # lens: (B,)
    
    image = de_normalize(image)
    B, C, H, W = image.shape
    label_len = cen_pos.shape[1]

    sum_image = image.sum(dim=[1, 2])
    # real_W = (sum_image>1e-3).argmax(dim=1).cpu().unsqueeze(1).unsqueeze(2)+1
    real_W = W

    if label is not None:
        temp = label.view(B * label_len)
        label_mask = temp.nonzero()[:, 0]
        cen_pos = cen_pos.view(B*label_len, 2)
        temp = torch.ones_like(cen_pos) * 2
        temp[label_mask, :] = cen_pos[label_mask, :]
        cen_pos = temp.view(B, label_len, 2)
        cen_pos = cen_pos.cpu().numpy()

    center_pos_y, center_pos_x = np.where(np.ones((H, W)) > 0)
    center_pos_y = np.expand_dims(center_pos_y, 0)
    center_pos_x = np.expand_dims(center_pos_x, 0)
    center_pos_y = np.expand_dims(center_pos_y, 2) # (1, H*W, 1)
    center_pos_x = np.expand_dims(center_pos_x, 2)

    # bb_center_x = np.expand_dims(cen_pos[:, :, 0], 1) # (B, 1, label_len)
    # bb_center_y = np.expand_dims(cen_pos[:, :, 1], 1)

    bb_center_x = np.expand_dims(cen_pos[:, :, 0], 1) # (B, 1, label_len)
    bb_center_y = np.expand_dims(cen_pos[:, :, 1], 1)


    # pdb.set_trace()
    # bb_center_x = (bb_center_x + 1) / 2 * real_W.cpu().numpy()
    bb_center_x = (bb_center_x + 1) / 2 * real_W
    bb_center_y = (bb_center_y + 1) / 2 * H

    ord_dis = (center_pos_x - bb_center_x) ** 2 + (center_pos_y - bb_center_y) ** 2
    ord_dis = np.exp(- ord_dis / (2 * sigma ** 2)) # (B, H*W, label_len)

    pos_map = torch.from_numpy(ord_dis).view(B*H*W, label_len).float()

    range_list = list(range(label_len))
    cmap = plt.cm.prism(range_list)[:, :3] # (label_len, 3)
    cmap = torch.from_numpy(cmap).float()
    temp = torch.matmul(pos_map, cmap)
    temp = temp.view(B, H, W, 3).permute([0, 3, 1, 2])
    vis_img = (image * 0.3 + temp * 0.7).clamp(0, 1)

    return split_batch(vis_img)


def get_image_prob_map(image):
    B, label_len, num_classes = image.shape
    minv = image.min()
    maxv = image.max()
    image = (image-minv) / (maxv - minv)
    image = image.unsqueeze(1)

    return split_batch(image)

def get_image_loc_map(image, loc_map, ratio=[0.6, 1]):
    image = de_normalize(image)
    B, H, W = loc_map.shape
    temp = loc_map.unsqueeze(1).repeat(1, 3, 1, 1)
    vis_img = torch.clamp((image * ratio[0] + temp * ratio[1]), 0, 1)

    return split_batch(vis_img, dim=1)

def get_image_mul_map(image, ord_map, loc_map, label_len):
    image = de_normalize(image)

    if len(ord_map.shape) == 4:
        ord_map = ord_map.max(1)[1]

    B, H, W = ord_map.shape

    ord_map_one_hot = torch.zeros(B, label_len, H, W)
    ord_map_one_hot.scatter_(1, ord_map.unsqueeze(1).long(), torch.ones(B, label_len, H, W))

    range_list = list(range(label_len))
    cmap = plt.cm.prism(range_list)[:, :3] # (label_len, 3)
    # cmap[0] = np.array([0, 0, 0])

    temp = ord_map_one_hot.permute([0, 2, 3, 1]).contiguous() # (B, H, W, label_len)
    temp = temp.view(-1, label_len)
    temp = torch.matmul(temp, torch.from_numpy(cmap).float())
    temp = temp.view(B, H, W, 3)
    temp = temp * loc_map.unsqueeze(3)
    temp = temp.permute([0, 3, 1, 2])

    vis_img = (image * 0.3 + temp * 0.7).clamp(0, 1)

    return split_batch(vis_img)


def get_image_seg_map(image, seg_map, num_classes, label_upsample=1, trans_background=True, dim=1):
    """
    Input:
        image: (B, 3, H, W)
        seg_map: (B, H, W)

    Output:
        vis_img: (3, B*H, W)
    """
    image = de_normalize(image)
    B, C, H, W = image.shape

    if label_upsample > 1:
        seg_map = seg_map.unsqueeze(dim=1)
        seg_map = F.interpolate(seg_map.float(), scale_factor=label_upsample).long()
        seg_map = seg_map.squeeze(dim=1)

    seg_map_one_hot = torch.zeros(B, num_classes, H, W)
    seg_map_one_hot.scatter_(1, seg_map.unsqueeze(1).long(), torch.ones(B, num_classes, H, W))

    range_list = list(range(num_classes))
    # random.shuffle(range_list)
    cmap = plt.cm.Set1(range_list)[:, :3] # (num_classes, 3)
    # if trans_background:
    #     cmap[1] = np.array([0, 0, 0])

    temp = seg_map_one_hot.permute([0, 2, 3, 1]).contiguous() # (B, H, W, num_classes)
    temp = temp.view(-1, num_classes)
    temp = torch.matmul(temp, torch.from_numpy(cmap).float())
    temp = temp.view(B, H, W, 3)
    temp = temp.permute([0, 3, 1, 2])

    vis_img = (image * 0.8 + temp * 0.2)


    return split_batch(vis_img, dim=dim)

def get_image_ord_map(image, ord_map):
    B, label_len, H, W = ord_map.shape
    img_H = image.shape[2]

    ord_map = ord_map.max(1)[1]
    
    if H == 1:
        ord_map = ord_map.repeat([1, img_H, 1])

    return get_image_seg_map(image, ord_map, label_len, True)


def get_image_text(image, pred_str):
    """
    Input:
        image: (B, C, H, W)
        pred_str: [str] * B

    Output:
        visualized_img: (H*B, W, C)
    """
    # data = torch.clamp(data, 0, 1) * 255
    # data = (data + 0.5) * 255
    # data = data.numpy().astype(np.uint8)

    # outputs = outputs.transpose(0, 1)

    image = de_normalize(image)

    B, C, H, W = image.shape
    assert C == 1 or C == 3

    img_list = []
    label_list = []
    for i in range(min(B, 16)):

        cur_str = pred_str[i]
        # visual_data = np.transpose(data[i], [1, 2, 0])
        visual_data = image[i]
        if C == 1:
            visual_data = visual_data[0, :, :]

        text_img = np.zeros((H, W))
        pos = (16, 16)
        temp = cur_str.split(',')
        for c in temp:
            text_img = cv2.putText(text_img, c, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (255, 255, 255), 1, bottomLeftOrigin=False)
            pos = (pos[0], pos[1] + 16)
        text_img = torch.from_numpy(text_img).float() / 255.0

        if C == 3:
            text_img = text_img.unsqueeze(0).repeat(3, 1, 1)

        visual_data = torch.cat([visual_data, text_img], dim=2)
        img_list.append(visual_data)
        img_list.append(torch.zeros(C, 10, W*2))

    img_visual = torch.cat(img_list, dim=1)

    return img_visual

def vis_weibull_models(weibull_models, dis_per_class, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(weibull_models)):
        if weibull_models[i] != []:
            x = np.linspace(0, 1, 1000)
            plt.plot(x, stats.exponweib.pdf(x, *weibull_models[i]))

            for j in range(len(dis_per_class)):
                plt.hist(dis_per_class[j][i], 30, density=True, alpha=0.5)

            plt.savefig(os.path.join(save_dir, 'class_{}.png'.format(i)))
            plt.close('all')
def vis_weibull_models(outlier_probs, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x = np.linspace(0, 1, 1000)

    for j in range(len(outlier_probs)):
        plt.hist(outlier_probs[j][i], 30, density=True, alpha=0.5)

    plt.savefig(os.path.join(save_dir, 'outlier_probs.png'))
    plt.close('all')
