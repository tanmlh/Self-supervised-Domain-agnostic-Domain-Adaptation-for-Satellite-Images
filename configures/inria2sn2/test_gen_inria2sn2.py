"""
testing configures
"""
# version = '1.5.3'
version = 'our_gen_v7.4'
num_iter = 100000
debug = False

import importlib
import os
# conf = importlib.import_module('configures.inria2sn2.train_inria2sn2_v{}'.format(version.replace('.', '_'))).conf
conf = importlib.import_module('configures.inria2sn2.{}'.format(version.replace('.', '_'))).conf

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = 'test_{}'.format(version.replace('.', '_'))
solver_conf['gpu_ids'] = '7'
solver_conf['metric_names'] = []
solver_conf['checkpoints_dir'] = './checkpoints/' + solver_conf['solver_name']
solver_conf['checkpoints_dir'] = './checkpoints/inria2sn2/' + solver_conf['solver_name']
# solver_conf['log_dir'] = './checkpoints/' + solver_conf['solver_name'] + '/logs'
solver_conf['log_dir'] = './checkpoints/inria2sn2/' + solver_conf['solver_name'] + '/logs'
solver_conf['load_state'] = True
# solver_conf['solver_state_path'] = './checkpoints/inria2sn2_v{}/network_{:0>6d}.pkl'.format(version, num_iter)
solver_conf['solver_state_path'] = './checkpoints/inria2sn2/{}/network_{:0>6d}.pkl'.format(version, num_iter)
solver_conf['phase'] = 'test'
solver_conf['load_epoch'] = False
solver_conf['save_freq'] = 1e8
solver_conf['print_freq'] = 100
solver_conf['save_img'] = False
solver_conf['num_iter'] = 10
solver_conf['plot'] = False
solver_conf['size_mat'] = 10
solver_conf['net_mode'] = 'eval'

loader_conf = conf['loader_conf']
loader_conf['dataset_name'] = 'inria2sn2_v2'
loader_conf['dataset_dir'] = '../../Datasets'

loader_conf['train_reader_conf'] = {}
# loader_conf['train_reader_conf']['file_path'] = '../../Datasets/Inria/train.txt'
loader_conf['train_reader_conf']['file_path'] = '../../Datasets/Inria/samples.txt'
loader_conf['train_reader_conf']['root_dir'] = '../../Datasets/Inria'
loader_conf['train_reader_conf']['dataset_type'] = 'mul_dom'
loader_conf['train_reader_conf']['sample_type'] = 'linear'

loader_conf['test_reader_conf'] = {}
# loader_conf['test_reader_conf']['file_path'] = '../../Datasets/DeepGlobe/SN2_train_filtered.txt'
loader_conf['test_reader_conf']['file_path'] = '../../Datasets/DeepGlobe/SN2_samples.txt'
loader_conf['test_reader_conf']['root_dir'] = '../../Datasets/DeepGlobe'
loader_conf['test_reader_conf']['dataset_type'] = 'mul_dom'
loader_conf['test_reader_conf']['sample_type'] = 'linear'

loader_conf['batch_size'] = 1
loader_conf['num_workers'] = 1
loader_conf['norm_type'] = 'plain'
loader_conf['random_type'] = 'linear'
loader_conf['A_sample_type'] = 'data12'
loader_conf['B_sample_type'] = 'data12'
# loader_conf['random_type'] = 'random'
# loader_conf['A_sample_type'] = 'data1'
# loader_conf['B_sample_type'] = 'data2'
loader_conf['phase'] = 'gen'
loader_conf['aug_conf'] = {'random_rotate': False, 'random_mirror': False, 'random_crop': False,
                           'base_size': [256, 256], 'crop_ratio': [0.5, 2.0], 'crop_size': [256, 256],
                           'resize': True, 'gaussian_blur': False}
# loader_conf['aug_conf'] = {'random_rotate': True, 'random_mirror': True, 'random_crop': True,
#                            'base_size': [256, 256], 'crop_ratio': [0.5, 2.0], 'crop_size': [256, 256],
#                            'resize': True, 'gaussian_blur': False}

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
eval_conf['reverse_color'] = False
eval_conf['plot_img_mat_v3'] = True

vis_conf = {}

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf,
        'eval_conf': eval_conf, 'vis_conf': vis_conf}
