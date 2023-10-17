"""
testing configures
"""
version = 'P.0.6'
num_iter = 160000
debug = False

import importlib
import os
conf = importlib.import_module('configures.mul_da.train_mul_da_v{}'.format(version.replace('.', '_'))).conf

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = 'mul_da_test_v{}'.format(version.replace('.', '_'))
solver_conf['gpu_ids'] = '7'
solver_conf['metric_names'] = []
solver_conf['checkpoints_dir'] = './checkpoints/' + solver_conf['solver_name']
solver_conf['log_dir'] = './checkpoints/' + solver_conf['solver_name'] + '/logs'
solver_conf['load_state'] = True
solver_conf['solver_state_path'] = './checkpoints/mul_da_v{}/network_{:0>6d}.pkl'.format(version, num_iter)
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
loader_conf['dataset_dir'] = '../../Datasets/mul_da/IAILD'
loader_conf['sampler_path'] = 'none'
# loader_conf['file_list_path'] = '../../Datasets/list/sample_17city.txt'
# loader_conf['file_list_path'] = '../../Datasets/list/sample_10city_v5_sr_2.txt'
loader_conf['file_list_path'] = '../../Datasets/mul_da/sample_inria_10city.txt'
# loader_conf['file_list_path'] = '../../Datasets/list/test_munich_sr.txt'
loader_conf['solver_name'] = solver_conf['solver_name']
loader_conf['batch_size'] = 1
loader_conf['use_augment'] = False
loader_conf['num_workers'] = 0
loader_conf['filter_size'] = 1
loader_conf['shuffle'] = False

## Network Options
net_conf = conf['net_conf']
net_conf['cls_idx'] = 1
net_conf['blocked_net'] = []
net_conf['feat_A_path'] = os.path.join(solver_conf['checkpoints_dir'], 'feat_A_path.npy')

eval_conf = {}
eval_conf['task_type'] = 'cls'
eval_conf['eval_type'] = 'batch'
eval_conf['log_dir'] = solver_conf['log_dir']
eval_conf['dataset_name'] = loader_conf['dataset_name']
eval_conf['num_classes'] = loader_conf['num_classes']
eval_conf['class_names'] = ['Compact high-rise', 'Compact mid-rise', 'Compact low-rise', 'Open high-rise',
                            'Open mid-rise', 'Open low-rise', 'Lightweight low-rise', 'Large low-rise',
                            'Sparsely built', 'Heavy industry', 'Dense trees', 'Scattered tree', 'Bush',
                            'scrub', 'Low plants', 'Bare rock or paved', 'Bare soil or sand', 'Water']

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf, 'eval_conf': eval_conf}
