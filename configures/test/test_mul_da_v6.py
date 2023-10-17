"""
testing configures
"""
version = 'Q.1.6'
num_iter = 200000
debug = False

import importlib
conf = importlib.import_module('configures.mul_da.train_mul_da_v{}'.format(version.replace('.', '_'))).conf

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = 'mul_da_test_v{}'.format(version.replace('.', '_'))
solver_conf['gpu_ids'] = '1'
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
solver_conf['num_iter'] = 1e9
solver_conf['plot'] = False
# solver_conf['plot_img_mat'] = True
solver_conf['plot_img_mat_v2'] = False
solver_conf['size_mat'] = 10
# solver_conf['test_type'] = 'whole'
solver_conf['test_type'] = 'patch'
solver_conf['net_mode'] = 'eval'


loader_conf = conf['loader_conf']
loader_conf['dataset_name'] = 'test_triplet'

loader_conf['dataset_dir'] = '../../Datasets'
# loader_conf['dataset_dir'] = '../../Datasets/mul_da_10'

loader_conf['file_list_str'] = '../../Datasets/list/{}'
# loader_conf['file_list_str'] = '../../Datasets/mul_da_10/list/{}.txt'
loader_conf['file_list_names'] = [
                                  'test_yaounde',
                                  'test_djibouti',
                                  'test_niamey',
                                  'test_thamaga',
                                  'test_daressalaam',
                                  'test_losangeles',
                                  'test_saopaulo',
                                  'test_lira',
                                  # 'test_jakarta',
                                  'test_newyork',
                                  'test_paris',
                                  'test_yaounde',
                                  'test_adelaide',
                                  'test_sydney',
                                  'test_rio'][:]

loader_conf['solver_name'] = solver_conf['solver_name']
loader_conf['batch_size'] = 1
loader_conf['use_augment'] = False
loader_conf['num_workers'] = 10
loader_conf['filter_size'] = 1
loader_conf['shuffle'] = False

loader_conf['patch_size'] = 256
loader_conf['pad_size'] = 0
loader_conf['img_H'] = 256
loader_conf['img_W'] = 256
loader_conf['step_size'] = 64
loader_conf['sampler_path'] = 'none'
loader_conf['patch_normalize'] = True

## Network Options
net_conf = conf['net_conf']
net_conf['cls_idx'] = 1
net_conf['blocked_net'] = []

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}
