import os
import torch
import torch.distributed as dist
import time
from models import model_dict, TrainTask

if __name__ == '__main__':
    start_time=time.time()
    # reference https://stackoverflow.com/questions/38050873/can-two-python-argparse-objects-be-combined/38053253
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    # dist.init_process_group(backend='nccl',
    #                         init_method='env://')
    # torch.cuda.set_device(dist.get_rank())
    model = MODEL(opt)
    model.fit()
    end_time=time.time()
    print("15个病人测试时间为：{:.6f}秒".format(end_time - start_time))
