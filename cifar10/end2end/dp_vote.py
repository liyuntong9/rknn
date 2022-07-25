import numpy as np
import torch
from torch.distributions.laplace import Laplace
import os
from stu_tools import get_logger, set_seed, create_config

from noiser import collision_simulator, rr_simulator
from copy import deepcopy

'''
中心/本地差分隐私 
数据特征存储在 /kolla/code/lyt/general_rknn/cifar10/end2end/feats 中
聚类中心的数量不用改动
结果存储在 /kolla/code/lyt/general_rknn/cifar10/end2end/feats/rr 或 collsion 或 cdp 中
'''
cfg = create_config()
set_seed(cfg.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

feats_dir = cfg.feats_dir
public_feats = np.load(os.path.join(feats_dir, 'public_feats.npy'))
private_feats = np.load(os.path.join(feats_dir, 'private_feats.npy'))
public_labels = np.load(os.path.join(feats_dir, 'public_labels.npy'))
private_labels = np.load(os.path.join(feats_dir, 'private_labels.npy'))
label_pred = np.load(os.path.join(feats_dir, 'label_pred.npy'))
center_indices = np.load(os.path.join(feats_dir, 'centroids.npy'))
cluster = cfg.cluster
neighbors = cfg.neighbors
eps_list = cfg.eps_list
exp_runs = cfg.exp_runs
dp_type = cfg.dp_type
num_cls = cfg.num_cls
type_dir = os.path.join(feats_dir, dp_type)
os.makedirs(type_dir, exist_ok=True)
center_indices = torch.from_numpy(center_indices)
private_feats = torch.from_numpy(private_feats)
public_feats = torch.from_numpy(public_feats)
public_labels = torch.from_numpy(public_labels)
private_labels = torch.from_numpy(private_labels)
label_pred = torch.from_numpy(label_pred)

center_labels = public_labels[center_indices]
center_feats = public_feats[center_indices]
private_samples = private_feats.size(0)
print(f'private_samples: {private_samples}')

for k in neighbors:
    counts = torch.zeros(cluster, num_cls)
    for i in range(private_samples):
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
        if dp_type == 'cdp':
            lap = Laplace(0, 2*1*k/eps)

        def addnoise(x):
            if dp_type == 'collision':
                return collision_simulator(eps, k, private_samples, x)
            elif dp_type == 'rr':
                return rr_simulator(eps, k, private_samples, x)
            elif dp_type == 'cdp':
                return x + lap.sample()
        for t in range(exp_runs):
            log_file = os.path.join(type_dir, f'log{cluster}_{k}_{eps_str}_{t + 1}.txt')
            log, log_close = get_logger(log_file)
            log(f'num_cluster: {cluster}, num_neighbor: {k}, eps: {eps}, num_runs: {t + 1}')
            noi_counts = deepcopy(counts).apply_(addnoise)
            center_pred_labels = noi_counts.max(dim=1)[1]


            np.save(os.path.join(type_dir, f'center_pred_labels{cluster}_{k}_{eps_str}_{t+1}.npy'), center_pred_labels.numpy())
            centers_acc = (center_pred_labels == center_labels).sum().item() / center_labels.size(0)
            log('centers acc: %.3f ' % centers_acc)
            label_results = label_pred.clone()
            for i in range(cluster):
                label_results[label_pred==i] = center_pred_labels[i]
            labels_acc = (label_results == public_labels).sum() / label_results.size(0)
            log('labels acc: %.3f' % labels_acc)
            np.save(os.path.join(type_dir, f'label_results{cluster}_{k}_{eps_str}_{t+1}.npy'), label_results.numpy())
            log_close()
