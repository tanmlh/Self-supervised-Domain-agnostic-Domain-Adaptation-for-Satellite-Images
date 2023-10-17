"""
testing configures
"""
version = '1.0.1'
num_iter = 200000
debug = False

import importlib
import os
conf = importlib.import_module('configures.lcz.train_lcz_v{}'.format(version.replace('.', '_'))).conf

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = 'lcz_gen_v{}'.format(version.replace('.', '_'))
solver_conf['gpu_ids'] = '1'
solver_conf['metric_names'] = []
solver_conf['checkpoints_dir'] = './checkpoints/' + solver_conf['solver_name']
solver_conf['log_dir'] = './checkpoints/' + solver_conf['solver_name'] + '/logs'
solver_conf['load_state'] = True
solver_conf['solver_state_path'] = './checkpoints/lcz_v{}/network_{:0>6d}.pkl'.format(version, num_iter)
solver_conf['phase'] = 'test'
solver_conf['load_epoch'] = False
solver_conf['save_freq'] = 1e8
solver_conf['print_freq'] = 100
solver_conf['save_img'] = False
solver_conf['num_iter'] = 10
solver_conf['plot'] = False
solver_conf['plot_img_mat_v3'] = True
solver_conf['size_mat'] = 10
solver_conf['net_mode'] = 'eval'

loader_conf = conf['loader_conf']
loader_conf['dataset_name'] = 'lcz42'
loader_conf['dataset_dir'] = '../../Datasets'
loader_conf['file_path'] = '../../Datasets/lcz42/m1483140/testing.h5'
loader_conf['batch_size'] = 1
loader_conf['num_workers'] = 0
loader_conf['norm_type'] = 'channel_wise'

## Network Options
net_conf = conf['net_conf']
net_conf['cls_idx'] = 1
net_conf['blocked_net'] = []
net_conf['feat_A_path'] = os.path.join(solver_conf['checkpoints_dir'], 'feat_A_path.npy')

eval_conf = {}
eval_conf['task_type'] = 'gen'
eval_conf['eval_type'] = 'batch'
eval_conf['log_dir'] = solver_conf['log_dir']
eval_conf['dataset_name'] = loader_conf['dataset_name']
eval_conf['num_classes'] = loader_conf['num_classes']

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf, 'eval_conf': eval_conf}
