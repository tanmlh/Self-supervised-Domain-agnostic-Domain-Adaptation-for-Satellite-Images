from utils import visualizer, evaluator
from utils import metric
import torch
import tqdm
import numpy as np
import cv2
import pdb
import os
from PIL import Image
import logging
import torch.nn.functional as F
import random

class Tester:

    def __init__(self, eval_conf):
        self.eval_conf = eval_conf
        self.eval_type = eval_conf['eval_type']
        self.dataset_name = eval_conf['dataset_name']
        self.num_classes = self.eval_conf['num_classes']

    def test_epoch_patch(self, solver, data_loader):

        eval = evaluator.get_evaluator(solver.eval_conf)

        solver.load_to_gpu()
        solver.net.eval()

        tq = tqdm.tqdm(data_loader)
        epoch_state = {}
        tq.set_description('Test on {} | Step: {}'.format(self.dataset_name, solver.global_step))

        for idx, batch in enumerate(tq):

            solver.set_tensors(batch)
            state = solver.net.forward(solver.tensors, 'test')

            pred_map = state['out|pred_map']
            label_map = state['out|label_map']

            eval.add_batch(label_map.view(-1).cpu().numpy(), pred_map.view(-1).cpu().numpy())

            if solver.global_step % solver.solver_conf['print_freq'] == 0:
                solver.summary_write_state(state, solver.global_step, 'test')

            solver.global_step += 1

            if 'num_iter' in solver.solver_conf and solver.global_step == solver.solver_conf['num_iter']:
                break

        eval.print_info()

    def get_patch_weight(solver):
        patch_size = solver.loader_conf['patch_size']
        choice = 1
        if choice == 0:
            step_size = (1.0 - 0.5)/(patch_size/2)
            a = np.arange(1.0, 0.5, -step_size)
            b = a[::-1]
            c = np.concatenate((b,a))
            ct = c.reshape(-1,1)
            x = ct*c
            return x
        elif choice == 1:
            min_weight = 0.5
            step_count = patch_size//4
            step_size = (1.0 - min_weight)/step_count
            a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
            a = a * min_weight
            for i in range(1, step_count + 1):
                a[i:-i, i:-i] += step_size
            a = cv2.GaussianBlur(a,(5,5),0)
            return a
        else:
            a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
            return a

    def test_epoch_whole(solver, data_loader):
        solver.load_to_gpu()

        img_H = solver.loader_conf['img_H']
        img_W = solver.loader_conf['img_W']
        patch_size = solver.loader_conf['patch_size']
        pad_size = solver.loader_conf['pad_size']
        step_size = solver.loader_conf['step_size']
        num_classes = solver.loader_conf['num_classes']
        cls_idx = solver.net_conf['cls_idx'] if 'cls_idx' in solver.net_conf else 0
        save_img = solver.solver_conf['save_img'] if 'save_img' in solver.solver_conf else False
        label_upsample = solver.solver_conf['label_upsample'] if 'label_upsample' in solver.solver_conf else 1

        weight = torch.tensor(get_patch_weight(solver))

        tq = tqdm.tqdm(data_loader)
        epoch_state = {}
        ious = []

        vis = visualizer.Visualizer()
        for idx, batch in enumerate(tq):
            tq.set_description('Test on {} | Step: {}'.format(solver.loader_conf['file_list_path'],
                                                              solver.global_step))
            img = batch['img_A']
            gt = batch['label_map_A']
            dom_A = batch['dom_A']

            pred_prob_map = torch.zeros(1, num_classes, img_H + pad_size * 2, img_W + pad_size * 2)
            for y in list(range(0, img_H + pad_size - patch_size, step_size)) + [img_H + pad_size - patch_size]:
                for x in list(range(0, img_W + pad_size - patch_size, step_size)) + [img_W + pad_size - patch_size]:
                    sub_batch = {}
                    sub_batch['img_A'] = img[:, :, y:y+patch_size, x:x+patch_size].cuda()
                    sub_batch['label_map_A'] = gt[:, y:y+patch_size, x:x+patch_size].cuda()
                    sub_batch['dom_A'] = dom_A

                    state = solver.process_batch(sub_batch, 'test', record=False)
                    sub_pred_map = state['else|cls_map'].cpu()

                    pred_prob_map[:, :, y:y+patch_size, x:x+patch_size] += sub_pred_map * weight

            pred_map = pred_prob_map.max(dim=1)[1]

            pred_map = pred_map[:, pad_size:pad_size+img_H, pad_size:pad_size+img_W]
            gt_map = gt[:, pad_size:pad_size+img_H, pad_size:pad_size+img_W]
            img = img[:, :, pad_size:pad_size+img_H, pad_size:pad_size+img_W]

            vis_pred_map = vis.get_image_seg_map(img, pred_map, num_classes, label_upsample=1)
            vis_gt_map = vis.get_image_seg_map(img, gt_map, num_classes, label_upsample=1)


            cur_iou = metric.cal_iou(pred_map, gt_map, num_classes, cls_idx=cls_idx)
            cur_iou = cur_iou.mean(dim=1).mean().item()
            ious.append(cur_iou)

            state = {'image|pred_map': vis_pred_map, 'image|label_map': vis_gt_map, 'scalar|miou': cur_iou}
            solver.summary_write_state(state, solver.global_step, 'test')

            tq.set_postfix({'iou': cur_iou})
            solver.global_step += 1

            if save_img:
                root_dir = solver.loader_conf['dataset_dir']
                save_dir = os.path.join(root_dir, solver.solver_conf['solver_name'])
                if not os.path.exists(os.path.join(save_dir, 'data')):
                    os.makedirs(os.path.join(save_dir, 'data'))
                img_name = batch['A_paths'][0][0].split('/')[-1]
                img_path = os.path.join(save_dir, 'data', img_name)
                pred_map = (pred_map[0].numpy() * 255).astype(np.uint8)

                if label_upsample > 1:
                    B, C, H, W = img.shape
                    pred_map = cv2.resize(pred_map, (int(W * label_upsample), int(H * label_upsample)))


                Image.fromarray(pred_map).save(img_path)
                file_list_path = solver.loader_conf['file_list_path'].split('/')[-1]
                file_list_path = os.path.join(save_dir, file_list_path)
                with open(file_list_path, 'a') as f:
                    img_path2 = os.path.join(solver.solver_conf['solver_name'], 'data', img_name)
                    f.write('{} {} {} {}\n'.format(img_path2,
                                                   batch['A_paths'][1][0],
                                                   batch['A_paths'][2][0],
                                                   batch['A_paths'][3][0]))


        print(sum(ious) / len(ious))

    def test_epoch(self, solver, data_loader):
        test_type = solver.solver_conf['test_type'] if 'test_type' in solver.solver_conf else 'patch'
        if test_type == 'patch':
            return self.test_epoch_patch(solver, data_loader)
        else:
            raise ValueError()
