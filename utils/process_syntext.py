import os, re
from scipy.io import loadmat 
import numpy as np
import pdb
import pickle
import cv2
from tensorboardX import SummaryWriter
import tqdm
import itertools
import time

summary_writer = SummaryWriter(log_dir='./test_log')

def cross_prod(v1, v2):
    # v1: (2,)
    return (v1[0] * v2[1] - v1[1] * v2[0])

def inside_word(wordBB, p):
    # wordBB: (2, 4)
    # p: (2,)
    if wordBB.shape[1] == 4:
        p_tile = np.tile(np.expand_dims(p, 1), [1, 4])
        vec1 = p_tile - wordBB
        vec2 = wordBB.copy()
        vec2[:, 0] -= wordBB[:, 3]
        vec2[:, 1] -= wordBB[:, 0]
        vec2[:, 2] -= wordBB[:, 1]
        vec2[:, 3] -= wordBB[:, 2]

        prods = vec1[0, :] * vec2[1, :] - vec1[1, :] * vec2[0, :]

        if (prods < 0).sum() == 4:
            return True
        else:
            return False
    else: # AABB in this case
        if p[0] >= wordBB[0, 0] and p[0] <= wordBB[0, 1] \
           and p[1] >= wordBB[1, 0] and p[1] <= wordBB[1, 1]:
            return True
        else:
            return False

def get_cen_pos(H, W, ori_char_bb_list, ori_label, num_lexicon, label_len):

    char_bb_list = ori_char_bb_list
    label = ori_label

    num_char = len(char_bb_list)

    bb_pos_x = np.zeros((num_char, 4))
    bb_pos_y = np.zeros((num_char, 4))
    label = np.array(label)

    for i, char_bb in enumerate(char_bb_list):

        bb_pos_x[i] = char_bb.transpose([1, 0])[:, 1]
        bb_pos_y[i] = char_bb.transpose([1, 0])[:, 0]


    bb_center_x = bb_pos_x.mean(axis=-1) # (num_char,)
    bb_center_y = bb_pos_y.mean(axis=-1)

    bb_center_x = np.expand_dims(bb_center_x, 0)
    bb_center_y = np.expand_dims(bb_center_y, 0)

    bb_center_x = np.expand_dims(bb_center_x[0], 1) / H * 2 - 1
    bb_center_y = np.expand_dims(bb_center_y[0], 1) / W * 2 - 1


    # bb_cen_pos = np.concatenate([bb_center_x, bb_center_y], axis=1) # (num_char, 2)
    bb_cen_pos = np.concatenate([bb_center_y, bb_center_x], axis=1) # (num_char, 2)
    bb_cen_pos = np.pad(bb_cen_pos, ((0, label_len - num_char), (0, 0)),
                        'constant', constant_values = 0)


    out = {}
    out['cen_pos'] = bb_cen_pos

    return out




def render_seg_ord_map(H, W, ori_char_bb_list, ori_label, num_lexicon, label_len, sigma,
                       shrunk_ratio=0, ord_map_mode='loc'):
    """
    char_bb_list = []
    label = []
    for i, x in enumerate(ori_label):
        if x != num_lexicon - 1: # special char
            char_bb_list.append(ori_char_bb_list[i])
            label.append(x)
    """

    char_bb_list = ori_char_bb_list
    label = ori_label

    num_char = len(char_bb_list)

    seg_map = np.zeros((H, W))
    ord_map = np.zeros((H, W))
    bb_pos_x = np.zeros((num_char, 4))
    bb_pos_y = np.zeros((num_char, 4))
    diff_bb_pos_x = np.zeros((num_char, 4))
    diff_bb_pos_y = np.zeros((num_char, 4))
    label = np.array(label)


    center_pos_x, center_pos_y = np.where(seg_map > -1)

    for i, char_bb in enumerate(char_bb_list):
        bb_pos_x[i] = char_bb.transpose([1, 0])[:, 1]
        bb_pos_y[i] = char_bb.transpose([1, 0])[:, 0]

    center_pos_x = np.expand_dims(center_pos_x, 1)
    center_pos_y = np.expand_dims(center_pos_y, 1)

    center_pos_x_ = np.expand_dims(center_pos_x, 2) # (H*W, 1, 1)
    center_pos_y_ = np.expand_dims(center_pos_y, 2)

    for i in range(4):
        diff_bb_pos_x[:, i] = bb_pos_x[:, (i+1)%4] - bb_pos_x[:, i]
        diff_bb_pos_y[:, i] = bb_pos_y[:, (i+1)%4] - bb_pos_y[:, i]

    bb_center_x = bb_pos_x.mean(axis=-1) # (num_char,)
    bb_center_y = bb_pos_y.mean(axis=-1)

    temp_x = np.expand_dims(bb_center_x, 1)
    temp_y = np.expand_dims(bb_center_y, 1)

    bb_pos_x = temp_x + (bb_pos_x - temp_x) * (1 - shrunk_ratio)
    bb_pos_y = temp_y + (bb_pos_y - temp_y) * (1 - shrunk_ratio)

    bb_pos_x = np.expand_dims(bb_pos_x, 0) # (1, num_char, 4)
    bb_pos_y = np.expand_dims(bb_pos_y, 0)

    diff_bb_pos_x = np.expand_dims(diff_bb_pos_x, 0) # (1, num_char, 4)
    diff_bb_pos_y = np.expand_dims(diff_bb_pos_y, 0)

    bb_center_x = np.expand_dims(bb_center_x, 0)
    bb_center_y = np.expand_dims(bb_center_y, 0)

    temp_x = (center_pos_x_ - bb_pos_x)
    temp_y = (center_pos_y_ - bb_pos_y)

    # (H*W, num_char, 4)
    cross_prods = temp_x * diff_bb_pos_y - temp_y * diff_bb_pos_x
    idxes, label_idxes = np.where((cross_prods > 0).sum(axis=-1) == 4)
    idx_r, idx_c = idxes // W, idxes % W
    seg_map[idx_r, idx_c] = label[label_idxes]

    ord_dis = (center_pos_x - bb_center_x) ** 2 + (center_pos_y - bb_center_y) ** 2
    ord_dis = np.exp(- ord_dis / (2 * sigma ** 2))

    ord_dis = ord_dis.reshape((H, W, num_char))
    ord_dis = np.pad(ord_dis, ((0, 0), (0, 0), (0, label_len - num_char)), 'constant', constant_values = 0)
    ord_dis = np.transpose(ord_dis, [2, 0, 1]) # (label_len, H, W)


    if ord_map_mode == 'seg':
        ord_map[idx_r, idx_c] = label_idxes + 1
    else:
        temp_z, temp_x, temp_y = np.where(ord_dis > 0.2)
        ord_map[temp_x, temp_y] = temp_z + 1

    bb_center_x = np.expand_dims(bb_center_x[0], 1) / H * 2 - 1
    bb_center_y = np.expand_dims(bb_center_y[0], 1) / W * 2 - 1


    bb_cen_pos = np.concatenate([bb_center_x, bb_center_y], axis=1) # (num_char, 2)
    bb_cen_pos = np.pad(bb_cen_pos, ((0, label_len - num_char), (0, 0)),
                        'constant',
                        constant_values = 0)


    out = {}
    out['seg_map'] = seg_map
    out['ord_map'] = ord_map
    out['loc_map'] = ord_dis.max(0)
    out['cen_pos'] = bb_cen_pos

    return out

def get_AABB(BB):

    AABB = np.zeros((2, 2))
    AABB[:, 0] = np.min(BB, axis=1).round()
    AABB[:, 1] = np.max(BB, axis=1).round()
    AABB[0, 0] = max(AABB[0, 0], 0)
    AABB[1, 0] = max(AABB[1, 0], 0)

    AABB = AABB.astype(np.int32)

    return AABB



def crop_image(img_path, BB):
    img = cv2.imread(img_path)
    H, W, C = img.shape

    AABB = get_AABB(BB)
    AABB[0, 1] = min(AABB[0, 1], W)
    AABB[1, 1] = min(AABB[1, 1], H)

    crop_img = img[AABB[1, 0]:AABB[1, 1]+1, AABB[0, 0]:AABB[0, 1]]

    return crop_img, AABB


if __name__ == '__main__':
    m = loadmat('/mnt/lustre/zhangfahong/Datasets/SynthText/gt.mat')
    charBB = m['charBB'][0]
    wordBB = m['wordBB'][0]
    img_names = m['imnames'][0]
    txt = m['txt'][0]

    img_list = []

    num_imgs = 500

    for i in range(num_imgs):
    # for i in range(len(img_names)):
        word_list = []
        txt_list = []
        for x in txt[i]:
           txt_list += re.split('\s+', x.strip())
        if len(wordBB[i].shape) == 2:
            wordBB[i] = np.expand_dims(wordBB[i], axis=2)
        assert(len(txt_list) == wordBB[i].shape[2])
        # pdb.set_trace()

        for j in range(wordBB[i].shape[2]):
            char_list = []
            for k in range(charBB[i].shape[2]):
                char_center = charBB[i][:, :, k].mean(axis=1)
                if inside_word(get_AABB(wordBB[i][:, :, j]), char_center):
                    # pdb.set_trace()
                    char_list.append(charBB[i][:, :, k])

            word_list.append((wordBB[i][:, :, j], char_list, txt_list[j]))

        img_list.append((img_names[i][0], word_list))


    root_dir = '/mnt/lustre/zhangfahong/Datasets/'
    load_dir = '/mnt/lustre/zhangfahong/Datasets/SynthText'
    save_dir = '/mnt/lustre/zhangfahong/Datasets/SynTextCharAnno'
    cnt = 0
    pkl_list = []
    for img_obj in tqdm.tqdm(img_list):
        img_name, word_list = img_obj

        for i, word in enumerate(word_list):
            try:
                img, AABB = crop_image(os.path.join(load_dir, img_name), word[0])
                char_bb_list = [(bb - np.tile(AABB[:, 0:1], [1, 4])) for bb in word[1]]
                is_fail = 1
                if len(char_bb_list) == len(word[2]):
                    is_fail = 0
                    img_save_dir = os.path.join('SynTextCharAnno', 'image')
                else:
                    img_save_dir = os.path.join('SynTextCharAnno', 'image_fail')


                if not os.path.exists(os.path.join(root_dir, img_save_dir)):
                    os.makedirs(os.path.join(root_dir, img_save_dir))

                img_save_path = os.path.join(root_dir, img_save_dir, '{:0>7d}.jpg'.format(cnt))
                if not os.path.exists(img_save_dir):
                    cv2.imwrite(img_save_path, img)

                if not is_fail:
                    pkl_list.append((os.path.join(img_save_dir, '{:0>7d}.jpg'.format(cnt)), word[2], char_bb_list))

            except Exception:
                err_cnt += 1
                print(err_cnt)

            # for char_bb in char_bb_list:
            #     cv2.rectangle(img, (char_bb[0, 0], char_bb[1, 0]), (char_bb[0, 2], char_bb[1, 2]),
            #                   (0, 255, 0), 1)
            # summary_writer.add_image('char_bb', np.transpose(img, [2, 0, 1]), global_step=cnt)
            cnt += 1

    # pdb.set_trace()
    pkl_save_path = open(os.path.join(save_dir, 'train_syn_text_{:d}.pkl'.format(num_imgs)), 'wb')
    pickle.dump(pkl_list, pkl_save_path)
    pkl_save_path.close()


