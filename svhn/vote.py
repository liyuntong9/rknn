import numpy as np
import torch
from torch.distributions.laplace import Laplace
import os
from tools import set_seed, get_logger, create_config

cfg = create_config('env.yml')
set_seed(cfg.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)



feats_dir = cfg.feats_dir
public_feats = np.load(os.path.join(feats_dir, 'public_feats.npy'))
private_feats = np.load(os.path.join(feats_dir, 'private_feats.npy'))
public_labels = np.load(os.path.join(feats_dir, 'public_labels.npy'))
private_labels = np.load(os.path.join(feats_dir, 'private_labels.npy'))

clusters = cfg.clusters
neighbors = cfg.neighbors
eps_list = cfg.eps_list
exp_runs = cfg.exp_runs


for c in clusters:
    clu_dir = os.path.join(feats_dir, f'clu_{c}')
    label_pred = np.load(os.path.join(clu_dir, f'label_pred{c}.npy'))
    center_indices = np.load(os.path.join(clu_dir, f'center_indices{c}.npy'))


    center_indices = torch.from_numpy(center_indices)
    private_feats = torch.from_numpy(private_feats)
    public_feats = torch.from_numpy(public_feats)
    public_labels = torch.from_numpy(public_labels)
    private_labels = torch.from_numpy(private_labels)
    label_pred = torch.from_numpy(label_pred)

    center_labels = public_labels[center_indices]
    center_feats = public_feats[center_indices]


    for k in neighbors:
        counts = torch.zeros(c, 10)
        for i in range(private_feats.size(0)):
            feats = private_feats[i]
            diff_feats = feats.view(1, -1) - center_feats
            dist = torch.norm(diff_feats, p=2, dim=1)
            min_k = torch.topk(dist, k, largest=False)[1]
            for j in min_k:
                counts[j][private_labels[i]] += 1

        for eps in eps_list:
            if str(eps).find('.') != -1:
                list_str = list(str(eps))
                list_str.pop(str(eps).find('.'))
                eps_str = ''.join(list_str)
            else:
                eps_str = str(eps)

            lap = Laplace(0, 2*1*k/eps)

            def addnoise(x):
                return x + lap.sample()
            for t in range(exp_runs):
                log, log_close = get_logger(os.path.join(clu_dir, f'vote{c}_{k}_{eps_str}_{t+1}.log'))
                log(f'num_cluster: {c}, num_neighbor: {k}, epsilon: {eps_str}, num_runs: {t+1}')
                noi_counts = counts.apply_(addnoise)
                center_pred_labels = noi_counts.max(dim=1)[1]
                np.save(os.path.join(clu_dir, f'center_pred_labels{c}_{k}_{eps_str}_{t+1}.npy'), center_pred_labels.numpy())

                centers_acc = (center_pred_labels == center_labels).sum().item() / center_labels.size(0)
                log('centers acc: %.3f ' % centers_acc)
                label_results = label_pred.clone()
                for i in range(c):
                    label_results[label_pred==i] = center_pred_labels[i]
                labels_acc = (label_results == public_labels).sum() / label_results.size(0)
                log('labels acc: %.3f' % labels_acc)
                np.save(os.path.join(clu_dir, f'label_results{c}_{k}_{eps_str}_{t+1}.npy'), label_results.numpy())
                log_close()
