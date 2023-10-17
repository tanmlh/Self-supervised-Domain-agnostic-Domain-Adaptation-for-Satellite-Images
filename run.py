import pdb
import os
import sys
sys.path.append(os.getcwd())
import importlib
import argparse
import torch
from torch.utils.data import DataLoader
# import warnings
# warnings.filterwarnings('ignore')

from datasets import dataset
from models import base_solver
# import utils.distributed as dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default='configures.xxx')
    parser.add_argument('--state_path', type=str, default=None)

    args = parser.parse_args()
    args.conf_path = args.conf_path.split('.')[0].replace('/', '.')

    if args.state_path is not None:
        solver = base_solver.get_solver_from_solver_state(args.state_path)
        solver_conf = solver.conf['solver_conf']
        loader_conf = solver.conf['loader_conf']

    else:
        ## Load configurations ##
        conf = importlib.import_module(args.conf_path).conf
        solver_conf = conf['solver_conf']
        loader_conf = conf['loader_conf']
        solver_path = solver_conf['solver_path']
        solver = importlib.import_module(solver_path).get_solver(conf)
        if solver_conf['load_state']:
            if type(solver_conf['solver_state_path']) == list:
                for solver_state_path in solver_conf['solver_state_path']:
                    solver.load_solver_state(torch.load(solver_state_path, map_location='cpu'))
            else:
                solver.load_solver_state(torch.load(solver_conf['solver_state_path'], map_location='cpu'))

    batch_size = loader_conf['batch_size']
    num_workers = loader_conf['num_workers']

    if solver_conf['phase'] == 'train':

        train_dataset = dataset.get_dataset(loader_conf, phase='train')

        sampler = None
        if 'sampler_path' in loader_conf and loader_conf['sampler_path'] != 'none':
            sampler = importlib.import_module(loader_conf['sampler_path']).get_sampler(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True if sampler is None else False,
                                  sampler=sampler,
                                  collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None,
                                  num_workers=num_workers, pin_memory=False)

        solver.train(train_loader, None)

    elif solver_conf['phase'] == 'test':


        if 'file_list_names' in loader_conf:
            eval_conf = conf['eval_conf']
            for file_name in loader_conf['file_list_names']:

                loader_conf['test_reader_conf']['file_path'] = loader_conf['file_list_str'].format(file_name)
                eval_conf['dataset_name'] = file_name

                print(loader_conf['file_list_str'].format(file_name))
                test_dataset = dataset.get_dataset(loader_conf, phase='test')

                sampler = None
                if 'sampler_path' in loader_conf and loader_conf['sampler_path'] != 'none':
                    sampler = importlib.import_module(loader_conf['sampler_path']).get_sampler(test_dataset)

                test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False,
                                         sampler=sampler,
                                         num_workers=num_workers, pin_memory=False)

                state = solver.test(test_loader)

        else:
            test_dataset = dataset.get_dataset(loader_conf, phase='test')
            sampler = None
            if 'sampler_path' in loader_conf and loader_conf['sampler_path'] != 'none':
                sampler = importlib.import_module(loader_conf['sampler_path']).get_sampler(test_dataset)

            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False,
                                     sampler=sampler,
                                     num_workers=num_workers, pin_memory=False)
            state = solver.test(test_loader)


    elif solver_conf['phase'] == 'val':

        test_dataset = dataset.get_dataset(loader_conf, phase='test')
        sampler = None
        if 'sampler_path' in loader_conf and loader_conf['sampler_path'] != 'none':
            sampler = importlib.import_module(loader_conf['sampler_path']).get_sampler(test_dataset)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 sampler=sampler,
                                 num_workers=num_workers, pin_memory=False)
        state = solver.validate(test_loader)




