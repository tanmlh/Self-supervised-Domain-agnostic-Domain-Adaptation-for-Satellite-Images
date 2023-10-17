from . import eval_cls, eval_seg

def get_evaluator(eval_conf):

    task_type = eval_conf['task_type']
    if task_type == 'seg':
        return eval_seg.Evaluator(eval_conf)
    elif task_type == 'cls':
        return eval_cls.Evaluator(eval_conf)

