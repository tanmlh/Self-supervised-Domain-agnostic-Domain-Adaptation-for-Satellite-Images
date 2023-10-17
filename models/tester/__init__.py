from . import tester_cls, tester_seg, tester_gen

def get_tester(eval_conf):

    if eval_conf['task_type'] == 'cls':
        return tester_cls.Tester(eval_conf)

    elif eval_conf['task_type'] == 'seg':
        return tester_seg.Tester(eval_conf)

    elif eval_conf['task_type'] == 'gen':
        return tester_gen.Tester(eval_conf)

