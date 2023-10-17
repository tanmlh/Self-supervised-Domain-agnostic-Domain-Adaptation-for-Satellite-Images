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
import matplotlib.cm as cm
import sklearn.cluster as cluster

class Tester:

    def __init__(self, eval_conf):
        self.eval_conf = eval_conf
        self.eval_type = eval_conf['eval_type']
        self.dataset_name = eval_conf['dataset_name']
        self.reverse_color = eval_conf['reverse_color'] if 'reverse_color' in eval_conf else False

    def get_img_mat_v3(self, solver, imgs_A, dom_As=[]):

        B, C, H, W = imgs_A.shape
        pad = 5
        img_mat = torch.zeros(3, (H+pad)*(B)-pad, (W+pad)*(B)-pad)

        for i in range(B):
            batch = {}
            batch['img_A1'] = imgs_A[i:i+1]
            batch['img_A2'] = imgs_A[i:i+1]
            for j in range(B):
                batch['img_B1'] = imgs_A[j:j+1]

                if len(dom_As) > 0:
                    batch['dom_B'] = dom_As[j]

                solver.set_tensors(batch)
                solver.net.eval()
                state = solver.net.forward(solver.tensors, 'test')

                img_i2j = state['else|img_A2B'][0] if i != j else imgs_A[i]
                if self.reverse_color: img_i2j = img_i2j[[2,1,0], :]
                # img_mat[:, (i+1)*(H+pad):(i+1)*(H+pad)+H, (j)*(W+pad):(j)*(W+pad)+W] = img_i2j
                img_mat[:, (i)*(H+pad):(i)*(H+pad)+H, (j)*(W+pad):(j)*(W+pad)+W] = img_i2j

        vis = visualizer.Visualizer(solver.vis_conf)
        img_mat = vis.vis_image(img_mat.unsqueeze(0))

        return img_mat

    def get_sim_mat(self, solver, imgs_A):

        B, C, H, W = imgs_A.shape
        pad = 5

        img_mat = torch.zeros(3, (H+pad)*(B)-pad, (W+pad)*(B)-pad)
        for i in range(B):
            batch = {}
            batch['img_A1'] = imgs_A[i:i+1]
            for j in range(B):
                batch['img_B1'] = imgs_A[j:j+1]

                solver.set_tensors(batch)
                solver.net.eval()

                sim_A_B = solver.net.forward(solver.tensors, 'dis_net').cpu().item()
                sim_A_B = (sim_A_B + 1) / 2
                color = np.array(cm.jet(sim_A_B)[:3]).reshape(1, 1, 3)
                color = np.ones((H, W, 3)) * color
                cv2.putText(img=color, text='{:.2f}'.format(sim_A_B), org=(50, 140), fontFace=3, fontScale=2,
                            color=(1,1,1), thickness=2)
                color = torch.tensor(color).permute(2, 0, 1)

                img_mat[:, (i)*(H+pad):(i)*(H+pad)+H, (j)*(W+pad):(j)*(W+pad)+W] = imgs_A[i] if i == j else color


        vis = visualizer.Visualizer(solver.vis_conf)
        img_mat = vis.vis_image(img_mat.unsqueeze(0))

        return img_mat

    def get_affinity(self, solver, imgs_A):

        B, C, H, W = imgs_A.shape
        pad = 5

        sim_mat = np.zeros((B, B))
        for i in range(B):
            batch = {}
            batch['img_A1'] = imgs_A[i:i+1]
            for j in range(B):
                batch['img_B1'] = imgs_A[j:j+1]

                solver.set_tensors(batch)
                solver.net.eval()

                sim_A_B = solver.net.forward(solver.tensors, 'dis_net').cpu().item()
                sim_A_B = (sim_A_B + 1) / 2
                sim_mat[i, j] = sim_A_B

        return sim_mat

    def test_generator(self, solver, data_loader):
        feat_A_path = solver.net_conf['feat_A_path']

        solver.load_to_gpu()
        if solver.solver_conf['net_mode'] == 'train':
            solver.net.train()
        else:
            solver.net.eval()

        tq = tqdm.tqdm(data_loader)
        epoch_state = {}
        feat_As = []
        img_As = []
        dom_As = []

        tq.set_description('Test on {} | Step: {}'.format(self.dataset_name, solver.global_step))

        for idx, batch in enumerate(tq):
            solver.set_tensors(batch)
            state = solver.net.forward(solver.tensors, 'test')

            feat_A = state['else|feat_A']
            img_A2B = state['else|img_A2B']

            feat_As.append(feat_A.cpu().numpy())
            img_As.append(batch['img_A1'].cpu().numpy())
            if 'dom_A' in batch: dom_As.append(batch['dom_A'].cpu())

            if solver.global_step % solver.print_freq == 0:
                solver.summary_write_state(state, solver.global_step, 'test_gen')


            solver.global_step += 1
            if 'num_iter' in solver.solver_conf and solver.global_step == solver.solver_conf['num_iter']:
                break


        # random.shuffle(img_As)
        img_As = torch.tensor(np.concatenate(img_As, axis=0))

        feat_As = np.array(feat_As)
        np.save(feat_A_path, feat_As.mean(axis=0))

        if 'plot_img_mat_v3' in self.eval_conf and self.eval_conf['plot_img_mat_v3'] is True:
            img_mat = self.get_img_mat_v3(solver, img_As[:solver.solver_conf['size_mat']], dom_As)
            solver.summary_write_state({'image|img_mat': img_mat}, solver.global_step, 'test')

        if 'plot_sim_mat' in self.eval_conf and self.eval_conf['plot_sim_mat'] is True:
            img_mat = self.get_sim_mat(solver, img_As[:solver.solver_conf['size_mat']])
            solver.summary_write_state({'image|sim_mat': img_mat}, solver.global_step, 'test')

        if 'save_cls_imgs' in self.eval_conf and self.eval_conf['save_cls_imgs'] is True:
            num_cluster = self.eval_conf['num_cluster']
            sim_mat = self.get_affinity(solver, img_As)
            cls_idx = cluster.SpectralClustering(num_cluster, affinity='precomputed').fit_predict(sim_mat)
            cls_idx = torch.tensor(cls_idx)

            for i in range(num_cluster):
                cls_imgs = img_As[cls_idx == i]
                cls_imgs = [cls_imgs[j] for j in range(cls_imgs.shape[0])]
                cur_state = {'save|cls_imgs_{}'.format(i): cls_imgs}
                solver.summary_write_state(cur_state, solver.global_step, 'test')



    def test_epoch(self, solver, data_loader):
        return self.test_generator(solver, data_loader)
