"""
testing configures
"""
version = '1.3.2'
num_iter = 1000000
debug = False

import importlib
conf = importlib.import_module('configures.lcz.train_lcz_v{}'.format(version.replace('.', '_'))).conf

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = 'lcz_test_v{}'.format(version.replace('.', '_'))
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
solver_conf['num_iter'] = 1e9

loader_conf = conf['loader_conf']
loader_conf['dataset_name'] = 'lcz42'
loader_conf['dataset_dir'] = '../../Datasets'
loader_conf['file_path'] = '../../Datasets/lcz42/m1483140/testing.h5'
loader_conf['img_H'] = 32
loader_conf['img_W'] = 32
loader_conf['num_channels'] = 10
loader_conf['solver_name'] = solver_conf['solver_name']
loader_conf['batch_size'] = 32
loader_conf['use_augment'] = False
loader_conf['num_workers'] = 0
loader_conf['A_sample_type'] = 'data2'
loader_conf['B_sample_type'] = 'random'

## Network Options
net_conf = conf['net_conf']
net_conf['cls_idx'] = 1
net_conf['blocked_net'] = []

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
