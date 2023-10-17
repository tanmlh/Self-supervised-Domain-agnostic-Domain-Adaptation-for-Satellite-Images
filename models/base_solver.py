import os
import pdb
import importlib
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import tqdm
import itertools
from . import tester
from utils import visualizer, evaluator
import cv2
import numpy as np
# import utils.distributed as dist

def _print_conf(conf):
    print('\n-----------------------------------------\n')
    print('solver configuration:')
    for key, value in conf['solver_conf'].items():
        print(key + ': ' + str(value))

    print('\n-----------------------------------------\n')
    print('net configuration:')
    for key, value in conf['net_conf'].items():
        print(str(key) + ': ' + str(value))

    print('\n-----------------------------------------\n')
    print('loader configuration:')
    for key, value in conf['loader_conf'].items():
        print(key + ': ' + str(value))
    print('\n-----------------------------------------\n')

def get_solver_from_solver_state(solver_state_path):
    solver_state = torch.load(open(solver_state_path, 'rb'), map_location='cpu')
    conf = solver_state['conf']
    solver_path = conf['solver_conf']['solver_path']
    solver_name = solver_path.split('.')[-1]
    solver = importlib.import_module(solver_path).get_solver(conf)
    solver.load_solver_state(solver_state)

    print('Sucessfully load solver state from {}!'.format(solver_state_path))
    _print_conf(conf)

    return solver

class BaseSolver:
    def __init__(self, conf):
        self.conf = conf
        self.solver_conf = conf['solver_conf']
        self.net_conf = conf['net_conf']
        self.loader_conf = conf['loader_conf']
        self.eval_conf = conf['eval_conf'] if 'eval_conf' in conf else None
        self.vis_conf = conf['vis_conf'] if 'vis_conf' in conf else None
        self.checkpoints_dir = self.solver_conf['checkpoints_dir']
        self.save_freq = self.solver_conf['save_freq']
        self.print_freq = self.solver_conf['print_freq']
        self.use_dist = self.solver_conf['use_dist']
        self.log_dir = self.solver_conf['log_dir']

        self.max_epoch = self.solver_conf['max_epoch']
        self.max_iter = self.solver_conf['max_iter']
        self.cur_epoch = 1
        self.global_step = 0
        self.init_dist()
        self.init_network()
        self.init_optimizer()
        self.init_best_checkpoint_settings()
        self.init_tensors()
        self.visualizer = visualizer.Visualizer(self.vis_conf)
        self.evaluator = evaluator.get_evaluator(self.eval_conf)

        if not self.use_dist or self.rank == 0:
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)
            _print_conf(conf)


    def init_dist(self):
        if self.use_dist:
            rank, world_size = dist.dist_init()
            bn_group_size = self.solver_conf['bn_group_size']
            if bn_group_size == 1:
                bn_group = None
            else:
                assert world_size % bn_group_size == 0
                bn_group = dist.simple_group_split(world_size, rank, world_size // bn_group_size)
            self.rank = rank
            self.world_size = world_size

    def init_network(self):
        """
        Initialize the network model
        """
        net_path = self.solver_conf['net_path']
        self.net = importlib.import_module(net_path).get_model(self.net_conf)


    def train(self, train_loader, val_loader=None):

        self.load_to_gpu()
        """
        Train and test the network model with the given train and test data loader
        """
        start_epoch = self.cur_epoch
        for self.cur_epoch in range(start_epoch, self.max_epoch + 1):

            train_state = self.process_epoch(train_loader, 'train')
            # self.update_checkpoint(self.global_step)

            if val_loader is not None:
                eval_state = self.process_epoch(val_loader, 'val')
                self.update_best_checkpoint(eval_state, self.global_step)


    def test(self, test_loader):
        self.load_to_gpu()

        eval_state = self.process_epoch(test_loader, 'test')
        return eval_state

    def validate(self, val_loader):
        self.load_to_gpu()

        state_path_list = self.solver_conf['val_state_list']
        print('Number of states to validate: {}'.format(len(state_path_list)))
        for i, state_path in enumerate(state_path_list):
            try:
                state = torch.load(state_path)
            except Exception:
                print('error loading state {}'.format(state_path))
                break
            self.load_solver_state(state, 'gpu')

            val_state = self.process_epoch(val_loader, 'val')
            temp = {}
            for key, val in val_state.items():
                temp[key+'_mean'] = val

            self.summary_write_state(temp, self.global_step)
            print(state_path)
            print(temp)



    def get_solver_state(self):
        state = {}

        net_state = self.net.module.state_dict()
        optimizer_state = {}
        for key, optimizer in self.optimizers.items():
            optimizer_state[key] = optimizer.state_dict()

        state['cur_epoch'] = self.cur_epoch
        state['conf'] = self.conf
        state['global_step'] = self.global_step
        state['net_state'] = net_state
        state['optimizer_state'] = optimizer_state

        return state

    def load_solver_state(self, state, location='cpu'):
        if self.solver_conf['load_epoch']:
            self.cur_epoch = state['cur_epoch'] + 1
            self.global_step = state['global_step']

        net_state = state['net_state']
        if 'blocked_net' in self.net_conf:
            temp = lambda x: sum([x[0].startswith(a) for a in self.net_conf['blocked_net']]) == 0
            net_state = dict(filter(temp, net_state.items()))

        if location == 'cpu':
            self.net.load_state_dict(net_state, strict=False)
        else:
            # self.net.module.load_state_dict(state['net_state'])
            self.net.module.load_state_dict(net_state, strict=False)

        try:
            for key, state in state['optimizer_state'].items():
                self.optimizers[key].load_state_dict(state)

        except ValueError:
            print('Fail to load optimizer state!')

    def update_checkpoint(self, global_step):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        path = os.path.join(self.checkpoints_dir, 'network_{:0>6d}.pkl'.format(global_step))
        solver_state = self.get_solver_state()
        torch.save(solver_state, open(path, 'wb'))

        # old_path_network = os.path.join(self.checkpoints_dir, 'network_' + str(cur_epoch-1) + '.pkl')
        # if os.path.isfile(old_path_network) and (cur_epoch) % 10 != 0:
        #     os.remove(old_path_network)

    def init_best_checkpoint_settings(self):
        self.metric_names = self.solver_conf['metric_names']
        self.metric_values = {}
        self.best_epoch = None

    def update_best_checkpoint(self, eval_state, cur_epoch):
        for metric_name in self.metric_names:
            if 'scalar|'+metric_name not in eval_state.keys():
                return
            cur_metric_value = eval_state['scalar|'+metric_name]
            if metric_name not in self.metric_values.keys() or cur_metric_value > self.metric_values[metric_name]:
                self.metric_values[metric_name] = cur_metric_value
                self.best_epoch = cur_epoch

                path = os.path.join(self.checkpoints_dir, 'network_best_' + metric_name + '.pkl')
                cur_solver_state = self.get_solver_state()
                torch.save(cur_solver_state, open(path, 'wb'))

    def process_epoch(self, data_loader, phase='train'):
        tq = tqdm.tqdm(data_loader) if not self.use_dist or self.rank == 0 else data_loader

        epoch_state = {}
        for idx, batch in enumerate(tq):

            if self.global_step >= self.max_iter + 1:
                break

            self.adjust_lr(self.global_step)
            if not self.use_dist or self.rank == 0:
                tq.set_description('{} | {} | Ep: {} | Step: {} | Lr: {:.6f}'.format(self.solver_conf['solver_name'],
                                                                                     phase, self.cur_epoch,
                                                                                     self.global_step, self.cur_lr))

            cur_state = self.process_batch(batch, phase)

            if self.global_step % self.print_freq == 0 or phase != 'train':
                self.summary_write_state(cur_state, self.global_step, phase)

            for key, value in cur_state.items():
                if key.split('|')[0] == 'scalar' or key.split('|')[0] == 'record':
                    if key not in epoch_state:
                        epoch_state[key] = []
                    epoch_state[key].append(cur_state[key])

            if self.global_step % self.save_freq == 0 and self.global_step != 0:
                self.update_checkpoint(self.global_step)

            postfix = {}
            for metric_name in self.metric_names:
                if 'scalar|'+metric_name not in cur_state.keys():
                    continue

                postfix[metric_name] = cur_state['scalar|'+metric_name]
            tq.set_postfix(postfix)
            self.global_step = self.global_step + 1

        if not self.use_dist or self.rank == 0:
            tq.close()

        for key, value in epoch_state.items():
            epoch_state[key] = sum(epoch_state[key]) / len(epoch_state[key])

        return epoch_state

    def summary_write_state(self, state, step, phase='train'):
        for key, value in state.items():
            prefix, name = key.split('|')
            if prefix == 'scalar':
                self.summary_writer.add_scalar(phase+'_'+name, value, global_step=step)
            elif prefix == 'image' and value is not None:
                self.summary_writer.add_image(phase+'_'+name, value, global_step=step)
            elif prefix == 'graph':
                self.summary_writer.add_graph(*value)
            elif prefix == 'figure':
                self.summary_writer.add_figure(phase+'_'+name, value, global_step=step)
            elif prefix == 'vis':
                vis_key, vis_data = self.visualizer.visualize(key, value)
                self.summary_writer.add_image(phase+'_'+name, vis_data, global_step=step)
            elif prefix == 'save':
                for idx, img in enumerate(value):
                    np_img = self.visualizer.tensor2np(img)
                    img_path = os.path.join(self.log_dir, '{}_{}_{}.png'.format(name, step, idx))
                    cv2.imwrite(img_path, (np_img * 255).astype(np.uint8))

    def adjust_lr(self, cur_step):

        for key, optimizer in self.optimizers.items():

            if key in self.solver_conf['lr_conf']:
                lr_conf = self.solver_conf['lr_conf'][key]
            else:
                lr_conf = self.solver_conf['lr_conf']

            init_lr = lr_conf['init_lr']
            cur_lr = init_lr
            if 'warm_up' in lr_conf:
                start_lr, end_lr, warm_up_steps = lr_conf['warm_up']
                if cur_step <= warm_up_steps:
                    cur_lr = cur_step / warm_up_steps * (end_lr - start_lr) + start_lr

            if lr_conf['decay_type'] == 'LUT':
                decay_base = lr_conf['decay_base']
                mul = 1
                for i, step_num in enumerate(lr_conf['decay_steps']):
                    if cur_step >= step_num:
                        mul = decay_base ** (i+1)
                        cur_lr = init_lr * mul

            elif lr_conf['decay_type'] == 'poly':
                max_iter = self.solver_conf['max_iter']
                poly_base = lr_conf['poly_base']
                cur_lr = (1 - cur_step / max_iter) ** poly_base * init_lr

            else:
                raise ValueError

            if cur_lr is not None:
                self.cur_lr = cur_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr

    def load_to_gpu(self):
        gpu_ids = self.solver_conf['gpu_ids']
        if gpu_ids == '-1':
            return
        torch.backends.cudnn.benchmark = True

        if self.use_dist:
            # self.net = dist.DistModule(self.net.cuda())
            self.net = dist.DistModule(self.net)
            self.net.cuda()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            self.net = nn.DataParallel(self.net).cuda()

        # for key, tensor in self.tensors.items():
        #     self.tensors[key] = tensor.cuda()


        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    def my_backward(self, loss):

        grad_clip = 'grad_clip' in self.net_conf and self.net_conf['grad_clip'] != -1
        if self.use_dist:
            loss = loss / self.world_size
            loss.backward()
            dist.reduce_gradients(self.net, False)
        else:
            loss.mean().backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.net_conf['grad_clip'])

    def test_batch(self, batch):
        return self.tester.test_batch(self, batch)

    def test(self, data_loader):
        self.tester = tester.get_tester(self.eval_conf)
        return self.tester.test_epoch(self, data_loader)

    def process_batch(self, batch, phase='train'):
        raise NotImplementedError

    def init_tensors(self):
        self.tensors = {}
        raise NotImplementedError

    def set_tensors(self, batch):
        raise NotImplementedError

    def init_optimizer(self):

        def peek(iterable):
            try:
                first = next(iterable)
            except StopIteration:
                return False, None
            return True, itertools.chain([first], iterable)

        optimizers = {}
        lr = 1e-4
        if hasattr(self.net, 'parameters_group'):
            for key, parameters in self.net.parameters_group().items():
                parameters = filter(lambda x: x.requires_grad, parameters)
                flag, parameters = peek(parameters)
                if not flag:
                    continue
                if self.solver_conf['optimizer_name'] == 'SGD':
                    optimizers[key] = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
                elif self.solver_conf['optimizer_name'] == 'Adam':
                    optimizers[key] = torch.optim.Adam(parameters, lr=lr, betas=(0.5, 0.999))
                else:
                    raise ValueError
            self.optimizers = optimizers
        else:
            parameters = filter(lambda x: x.requires_grad, self.net.parameters())
            if self.solver_conf['optimizer_name'] == 'SGD':
                optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            elif self.solver_conf['optimizer_name'] == 'Adam':
                optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.5, 0.999))
            else:
                raise ValueError

            self.optimizers = {'default': optimizer}
