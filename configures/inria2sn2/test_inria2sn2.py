"""
testing configures
"""
version = 'dada'
# version = '1.4.4'
num_iter = 300000
debug = False
dataset_root = '../../Datasets/DeepGlobe/'

import importlib
import os
# conf = importlib.import_module('configures.inria2sn2.train_inria2sn2_v{}'.format(version.replace('.', '_'))).conf
# conf = importlib.import_module('configures.inria2sn2.{}'.format(version.replace('.', '_'))).conf
conf = importlib.import_module(f'configures.inria2sn2.{version}').conf

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = f'inria2sn2_test_{version}'
solver_conf['gpu_ids'] = '0'
solver_conf['metric_names'] = []
solver_conf['checkpoints_dir'] = './checkpoints/' + solver_conf['solver_name']
solver_conf['log_dir'] = './checkpoints/' + solver_conf['solver_name'] + '/logs'
solver_conf['load_state'] = True
solver_conf['solver_state_path'] = './checkpoints/inria2sn2/{}/network_{:0>6d}.pkl'.format(version, num_iter)
solver_conf['phase'] = 'test'
solver_conf['load_epoch'] = False
solver_conf['save_freq'] = 1e8
solver_conf['print_freq'] = 150
solver_conf['save_img'] = False
solver_conf['num_iter'] = 1e9
solver_conf['plot'] = False
solver_conf['net_mode'] = 'eval'

loader_conf = conf['loader_conf']
loader_conf['dataset_name'] = 'inria2sn2_v2'
loader_conf['dataset_dir'] = '../../Datasets'
loader_conf['file_list_names'] = ['Vegas', 'Paris', 'Shanghai', 'Khartoum', 'all'][:4]
loader_conf['file_list_str'] = f'{dataset_root}' + '/SN2_test_{}.txt'

loader_conf['test_reader_conf'] = {}
loader_conf['test_reader_conf']['file_path'] = '{dataset_root}/SN2_test.txt'
loader_conf['test_reader_conf']['root_dir'] = dataset_root
loader_conf['test_reader_conf']['dataset_type'] = 'mul_dom'
loader_conf['test_reader_conf']['sample_type'] = 'linear'

loader_conf['batch_size'] = 1
loader_conf['num_workers'] = 4
loader_conf['norm_type'] = 'plain'
loader_conf['random_type'] = 'linear'
loader_conf['A_sample_type'] = 'data2'
loader_conf['B_sample_type'] = 'data2'
loader_conf['use_hist_mat'] = False
loader_conf['reverse_color'] = True
loader_conf['aug_conf'] = {'random_rotate': False, 'random_mirror': False, 'random_crop': False,
                           'base_size': [640, 640], 'crop_ratio': [0.5, 2.0], 'crop_size': [640, 640],
                           # 'base_size': [512, 512], 'crop_ratio': [0.5, 2.0], 'crop_size': [512, 512],
                           'resize': True, 'gaussian_blur': False}

## Network Options
net_conf = conf['net_conf']

eval_conf = {}
eval_conf['task_type'] = 'seg'
eval_conf['eval_type'] = 'batch'
eval_conf['log_dir'] = solver_conf['log_dir']
eval_conf['dataset_name'] = loader_conf['dataset_name']
eval_conf['num_classes'] = loader_conf['num_classes']
eval_conf['reverse_color'] = False
eval_conf['class_names'] = ['Others',
                            'Buildings']

vis_conf = {}
vis_conf['colors'] = [[0,0,0], [255,255,255]]

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf,
        'eval_conf': eval_conf, 'vis_conf': vis_conf}
