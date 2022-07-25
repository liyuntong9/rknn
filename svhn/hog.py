
from torchvision.datasets import SVHN
from torch.utils.data import Subset, ConcatDataset
import numpy as np
import pickle
import os
from skimage.color import rgb2gray
from skimage.feature import hog
from tools import create_config, set_seed


cfg = create_config('env.yml')
set_seed(cfg.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
data_dir = cfg.data_dir
hog_dir = cfg.hog_dir
feats_dir = cfg.feats_dir



def save_hog(data, file):
    gray_data = [rgb2gray(i) for i in data]
    hog_data = np.array([hog(i, orientations=6, block_norm='L2') for i in gray_data], dtype = np.float32)
    with open(file, 'wb') as f:
        pickle.dump(hog_data, f)


def extract_feature(train_img, test_img):
    train_img = [np.asarray(data) for data in train_img]
    test_img = [np.asarray(data) for data in test_img]
    train_data = None
    each_length = int((9 + len(train_img)) / 10)
    for idx in range(10):
        hog_file = os.path.join(hog_dir, 'train_hog%d.pkl' % idx)
        if not os.path.exists(hog_file):
            p1 = idx * each_length
            p2 =min((idx + 1) * each_length, len(train_img))
            save_hog(train_img[p1:p2], hog_file)

        with open(hog_file, 'rb') as f:
            if train_data is not None:
                train_data = np.vstack((train_data, pickle.load(f)))
            else:
                train_data = pickle.load(f)
    hog_file = os.path.join(hog_dir, 'test_hog.pkl')
    if not os.path.exists(hog_file):
        save_hog(test_img, hog_file)
    with open(hog_file, 'rb') as f:
        test_data = pickle.load(f)
    return train_data, test_data




public_dataset = SVHN(root=data_dir, split='test', download=True)
private_dataset = SVHN(root=data_dir, split='train', download=True)
extra_dataset = SVHN(root=data_dir, split='extra', download=True)
print(len(public_dataset), len(private_dataset), len(extra_dataset))
private_dataset = ConcatDataset([private_dataset, extra_dataset])


rand_indices = np.arange(len(public_dataset))
np.random.shuffle(rand_indices)
public_dataset = Subset(public_dataset, rand_indices[:-1000])
print(len(private_dataset), len(public_dataset))

public_data, public_label, private_data, private_label = [], [], [], []
for data, label in public_dataset:
    public_data.append(data)
    public_label.append(label)
for data, label in private_dataset:
    private_data.append(data)
    private_label.append(label)
print(len(public_data), len(public_label), len(private_data), len(private_label))
private_data, public_data = extract_feature(private_data, public_data)
print(public_data.shape, private_data.shape)

public_label, private_label = np.array(public_label), np.array(private_label)

np.save(os.path.join(feats_dir, 'private_feats.npy'), private_data)
np.save(os.path.join(feats_dir, 'private_labels.npy'), private_label)
np.save(os.path.join(feats_dir, 'public_feats.npy'), public_data)
np.save(os.path.join(feats_dir, 'public_labels.npy'), public_label)