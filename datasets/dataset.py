from datasets import triplet_dataset_v2, test_triplet_dataset
from datasets import triplet_dataset_sr, test_triplet_dataset_sr
from datasets import lcz42_dataset, isprs_dataset, inria2sn2_dataset, inria2sn2_dataset_v2

def get_dataset(loader_conf, phase):
    if loader_conf['dataset_name'] == 'toy_mul_src':
        return toy_mul_src_dataset.ToyMulSrcDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'mul_src':
        return mul_src_dataset.MulSrcDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'mul_src_test':
        return mul_src_test_dataset.MulSrcTestDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'sin_src':
        return sin_src_dataset.SinSrcDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'raw_seg':
        return raw_seg_dataset.RawSegDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'iail':
        return iail_dataset.IAILDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'triplet':
        return triplet_dataset.TripletDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'triplet_v2':
        return triplet_dataset_v2.TripletDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'test_triplet':
        return test_triplet_dataset.TestTripletDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'triplet_sr':
        return triplet_dataset_sr.TripletDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'test_triplet_sr':
        return test_triplet_dataset_sr.TestTripletDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'pair_sr':
        return pair_dataset_sr.PairDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'pair':
        return pair_dataset.PairDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'test_pair_sr':
        return test_pair_dataset_sr.PairDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'lcz42':
        return lcz42_dataset.LCZ42Dataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'isprs':
        return isprs_dataset.ISPRSDataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'inria2sn2':
        return inria2sn2_dataset.INRIA2SN2Dataset(loader_conf, phase)
    elif loader_conf['dataset_name'] == 'inria2sn2_v2':
        return inria2sn2_dataset_v2.INRIA2SN2Dataset(loader_conf, phase)
    else:
        raise ValueError('No such dataset!')
