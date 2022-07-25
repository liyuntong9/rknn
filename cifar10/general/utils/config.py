import os
import yaml
from easydict import EasyDict
from .utils import mkdir_if_missing

def create_config(config_file_env, config_file_exp):
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()

    for k, v in config.items():
        cfg[k] = v


    simclr_dir = os.path.join(root_dir, 'simclr')
    feats_dir = os.path.join(root_dir, 'feats')


    mkdir_if_missing(simclr_dir)
    mkdir_if_missing(feats_dir)
    cfg['feats_dir'] = feats_dir
    cfg['data_dir'] = '/kolla/code/lyt/general_rknn/datasets'
    cfg['simclr_dir'] = simclr_dir
    cfg['simclr_checkpoint'] = os.path.join(simclr_dir, 'checkpoint.pth.tar')

    return cfg 
