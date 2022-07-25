
import torch
import numpy as np
from sklearn.cluster import KMeans
import os
from tools import set_seed, create_config
cfg = create_config('env.yml')
set_seed(cfg.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
feats_dir = cfg.feats_dir
public_feats = np.load(os.path.join(feats_dir, 'public_feats.npy'))
private_feats = np.load(os.path.join(feats_dir, 'private_feats.npy'))
public_labels = np.load(os.path.join(feats_dir, 'public_labels.npy'))
private_labels = np.load(os.path.join(feats_dir, 'private_labels.npy'))

clusters = cfg.clusters

for c in clusters:
    clu_dir = os.path.join(feats_dir, f'clu_{c}')
    os.makedirs(clu_dir, exist_ok=True)

    estimator = KMeans(n_clusters=c, init='k-means++', algorithm='auto')
    estimator.fit(public_feats)
    label_pred = estimator.labels_

    public_feats = torch.from_numpy(public_feats)
    center_feats = torch.from_numpy(estimator.cluster_centers_)
    label_pred = torch.from_numpy(label_pred).type(torch.int64)
    center_indices = []
    index_range = torch.arange(public_feats.size(0))

    for i in range(c):
        mask = label_pred == i
        masked_feats = public_feats[mask]
        dist = masked_feats - center_feats[i]
        dist = torch.norm(dist, p=2, dim=-1)
        idx = dist.min(dim=0)[1]
        center_indices.append(index_range[mask][idx])

    np.save(os.path.join(clu_dir, f'center_indices{c}.npy'), np.array(center_indices))
    np.save(os.path.join(clu_dir, f'label_pred{c}.npy'), label_pred.numpy())
    np.save(os.path.join(clu_dir, f'center_feats{c}.npy'), center_feats.numpy())
