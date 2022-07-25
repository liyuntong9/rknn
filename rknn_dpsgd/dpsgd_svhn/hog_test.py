

from torchvision.datasets import SVHN
from torch.utils.data import Subset
import numpy as np
import pickle
import os
from skimage.color import rgb2gray
from skimage.feature import hog
import torch
import random
seed = 42 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def save_hog(data, file):
    gray_data = [rgb2gray(i) for i in data]
    hog_data = np.array([hog(i, orientations=6, block_norm='L2') for i in gray_data], dtype = np.float32)
    with open(file, 'wb') as f:
        pickle.dump(hog_data, f)


def extract_feature(test_img):
    test_img = [np.asarray(data) for data in test_img]
    hog_file = os.path.join('svhn_hog', 'test_hog.pkl')
    if not os.path.exists(hog_file):
        save_hog(test_img, hog_file)
    with open(hog_file, 'rb') as f:
        test_data = pickle.load(f)
    return test_data


test_dataset = SVHN(root='/private/general_rknn/datasets', split='test', download=True)

rand_indices = np.arange(len(test_dataset))
np.random.shuffle(rand_indices)
test_dataset = Subset(test_dataset, rand_indices[-1000:])

test_data, test_label = [], []
for data, label in test_dataset:
    test_data.append(data)
    test_label.append(label)

print(len(test_data), len(test_label))
test_data = extract_feature(test_data)
print(test_data.shape)

test_data, test_label = np.array(test_data), np.array(test_label)

np.save(os.path.join('svhn_hog', 'test_data.npy'), test_data)
np.save(os.path.join('svhn_hog', 'test_label.npy'), test_label)


