import torch
import numpy as np
import os
from torch.distributions.laplace import Laplace
from torchvision import datasets
from tools import set_seed, get_logger, create_config, to_str
from copy import deepcopy
cfg = create_config()
set_seed(cfg.seed)
root_dir = cfg.root_dir
data_dir = cfg.data_dir
clusters = cfg.clusters
eps_list = cfg.eps_list
neighbors = cfg.neighbors
exp_runs = cfg.exp_runs
public_samples = 5000
private_ds = datasets.MNIST(data_dir, train=True)
public_ds = datasets.MNIST(data_dir, train=False)
indices = np.random.choice(range(len(public_ds)), public_samples, replace=False)
public_ds.data = public_ds.data[indices]
public_ds.targets = public_ds.targets[indices]
num_cls = cfg.num_cls
for c in clusters:
    ass_path = os.path.join(root_dir,f'clu_{c}')
    clu_ass = np.load(os.path.join(ass_path, 'public_ass{}.npy'.format(c)))

    for k in neighbors:
        vote_ass = np.load(os.path.join(ass_path, 'private_ass{}_{}.npy'.format(c, k)))
        counts = np.zeros([c, num_cls])
        one_hot = np.eye(num_cls)

        for i in range(c):
            mask = (vote_ass == i)[:, 0]
            for j in range(1, k):
                mask |= (vote_ass == i)[:, j]
            counts[i, :] = one_hot[private_ds.targets[mask]].sum(axis=0)

        for eps in eps_list:
            eps_str = to_str(eps)
            lap = Laplace(0, 2*1*k/eps)
            def add_noise(x):
                return x + lap.sample()
            for t in range(exp_runs):
                log_file = os.path.join(ass_path, f'log{c}_{k}_{eps_str}_{t + 1}.txt')
                log, log_close = get_logger(log_file)
                log(f'num_cluster: {c}, num_neighbor: {k}, eps: {eps}, num_runs: {t+1}')
                noi_counts = torch.from_numpy(deepcopy(counts)).apply_(add_noise)
                noi_max = noi_counts.argmax(1).numpy()
                label_results = np.zeros(public_samples, dtype=np.int64)
                for i in range(c):
                    mask = clu_ass == i
                    label_results[mask] = noi_max[i]
                label_acc = sum(label_results == public_ds.targets.numpy()) / public_samples
                log('label acc: %.4f' % label_acc )
                np.save(os.path.join(ass_path, 'label_results{}_{}_{}_{}.npy'.format(c, eps_str, k, t+1)), label_results)
