import argparse
import json
import os
from learn import Trainer
from utils import create_config, ConfigMapper
def main():
    parser = argparse.ArgumentParser(description='Semi-supervised Learning')
    parser.add_argument('--cfg_path', type=str, default='train.json')
    args = parser.parse_args()

    with open(args.cfg_path, "r") as f:
        configs = json.load(f)
    env = create_config('../env.yml')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(env.gpu_id)
    clusters = env.clusters
    neighbors = env.neighbors
    eps_list = env.eps_list
    exp_runs = env.exp_runs
    for k, v in env.items():
        configs[k] = v

    arg_dict = vars(args)
    for key in arg_dict:
        if key in configs:
            if arg_dict[key] is not None:
                configs[key] = arg_dict[key]
    configs = ConfigMapper(configs)
    for c in clusters:
        for k in neighbors:
            for eps in eps_list:
                for t in range(exp_runs):
                    if str(eps).find('.') != -1:
                        list_str = list(str(eps))
                        list_str.pop(str(eps).find('.'))
                        eps_str = ''.join(list_str)
                    else:
                        eps_str = str(eps)
                    configs.c = c
                    configs.k = k
                    configs.eps_str = eps_str
                    configs.t = t+1

                    configs.clu_dir = os.path.join(configs.feats_dir, f'clu_{c}')
                    configs.out_dir = f'{c}_{k}_{eps_str}_{t+1}_out'
                    configs.log_dir = f'{c}_{k}_{eps_str}_{t+1}_log'
                    labeled_idx_file = os.path.join(configs.clu_dir, f'center_indices{c}.npy')
                    label_file = os.path.join(configs.clu_dir, f'center_pred_labels{c}_{k}_{eps_str}_{t+1}.npy')

                    if configs.mode == 'train':
                        MixmatchTrainer = Trainer(configs, labeled_idx_file, label_file)
                        print('init done!')
                        MixmatchTrainer.train()
                    elif configs.mode == 'eval':
                        pass
                    else:
                        raise ValueError("Invalid mode, ['learn', 'eval'] modes are supported")


if __name__ == '__main__':

    main()
