import torch
import numpy as np
import random
import os
from easydict import EasyDict
import yaml

def create_config(yml_file):
    with open(yml_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v
    return cfg

def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_logger(log_filename, display=True):
    f = open(log_filename, 'w')
    counter = [0]
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return logger, f.close
