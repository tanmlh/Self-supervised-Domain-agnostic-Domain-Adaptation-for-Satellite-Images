"""
testing configures
"""
model_name='mul_da'
# model_name='raw_seg'
version = 'P.2.3'
num_iter = 300000
debug = False

import importlib
conf = importlib.import_module('configures.{}.train_{}_v{}'.format(model_name, model_name, version.replace('.', '_'))).conf

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = 'mul_da_test_v{}'.format(version.replace('.', '_'))
solver_conf['gpu_ids'] = '2' # set '-1' to disable gpu
solver_conf['metric_names'] = []
solver_conf['checkpoints_dir'] = './checkpoints/' + solver_conf['solver_name']
solver_conf['log_dir'] = './checkpoints/' + solver_conf['solver_name'] + '/logs'
solver_conf['load_state'] = True
solver_conf['solver_state_path'] = './checkpoints/{}_v{}/network_{:0>6d}.pkl'.format(model_name, version, num_iter)
solver_conf['phase'] = 'test'
solver_conf['load_epoch'] = False
solver_conf['save_freq'] = 1e8
solver_conf['print_freq'] = 10
solver_conf['save_img'] = True
solver_conf['num_iter'] = 1e8
solver_conf['plot'] = False
solver_conf['plot_img_mat'] = False
solver_conf['test_type'] = 'whole'
solver_conf['net_mode'] = 'eval'
solver_conf['label_upsample'] = 1


loader_conf = conf['loader_conf']
loader_conf['shuffle'] = False
# loader_conf['dataset_name'] = 'mul_src_test'
# loader_conf['dataset_name'] = 'sin_src'

loader_conf['dataset_name'] = 'raw_seg'
loader_conf['dataset_dir'] = '../../Datasets/mul_da/IAILD'
loader_conf['file_list_path'] = '../../Datasets/mul_da/IAILD/AerialImageDataset/test.txt'
loader_conf['patch_size'] = 512
loader_conf['pad_size'] = 512
loader_conf['step_size'] = 256
loader_conf['sampler_path'] = 'none'
loader_conf['use_resize'] = True
loader_conf['img_H'] = 5000
loader_conf['img_W'] = 5000

# loader_conf['dataset_dir'] = '../../Datasets/mul_da/IAILD'
# loader_conf['file_list_path'] = '../../Datasets/mul_da/IAILD/IAILD_splited_512_256/test_part_v1_1000.txt'
# loader_conf['dataset_dir'] = '../../Datasets/mul_da'
# loader_conf['file_list_path'] = '../../Datasets/mul_da/5citybasemap/all_5city_v3.txt'
# loader_conf['file_list_path'] = '../../Datasets/mul_da/africabasemap/test_africa_{}.txt'.format(test_name[0])
# loader_conf['file_list_str'] = '../../Datasets/mul_da/{}.txt'
# loader_conf['file_list_names'] = ['5citybasemap/test_5city_v3',
#                                   'africabasemap/test_africa_casablanca',
#                                   'africabasemap/test_africa_daressalaam',
#                                   'africabasemap/test_africa_niamey',
#                                   'africabasemap/test_africa_nairobi']

loader_conf['solver_name'] = solver_conf['solver_name']
loader_conf['batch_size'] = 1
loader_conf['use_augment'] = False
loader_conf['num_workers'] = 1
loader_conf['filter_size'] = 1

## Network Options
net_conf = conf['net_conf']
net_conf['cls_idx'] = 1

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}
