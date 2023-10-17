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

    def test_epoch_patch(self, solver, data_loader):

        eval = evaluator.get_evaluator(solver.eval_conf)
        num_classes = self.eval_conf['num_classes']

        solver.load_to_gpu()
        solver.net.eval()

        tq = tqdm.tqdm(data_loader)
        epoch_state = {}
        tq.set_description('Test on {} | Step: {}'.format(self.dataset_name, solver.global_step))

        for idx, batch in enumerate(tq):

            solver.set_tensors(batch)
            state = solver.net.forward(solver.tensors, 'test')

            preds = state['out|preds']
            labels = state['out|labels']

            eval.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

            if solver.global_step % solver.solver_conf['print_freq'] == 0:
                solver.summary_write_state(state, solver.global_step, 'test')

            solver.global_step += 1

            if 'num_iter' in solver.solver_conf and solver.global_step == solver.solver_conf['num_iter']:
                break

        eval.print_info()

    def test_epoch(self, solver, data_loader):
        test_type = solver.solver_conf['test_type'] if 'test_type' in solver.solver_conf else 'patch'
        if test_type == 'patch':
            return self.test_epoch_patch(solver, data_loader)
        else:
            raise ValueError()
