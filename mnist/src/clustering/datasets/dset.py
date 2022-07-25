from abc import ABCMeta
import numpy as np
from torch.utils.data.dataset import Dataset, ConcatDataset
from torchvision.datasets import MNIST, USPS
from torchvision.transforms import ToTensor, Compose
from utils import use_seed
from utils.path import DATASETS_PATH


class _AbstractDataset(Dataset):
    __metaclass__ = ABCMeta
    root = DATASETS_PATH

    dataset_class = NotImplementedError
    name = NotImplementedError
    n_classes = NotImplementedError
    n_channels = NotImplementedError
    img_size = NotImplementedError
    label_shift = 0
    n_samples = None

    def __init__(self, split):
        super().__init__()
        if self.name == 'mnist':
            if split == 'public':
                dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, train=False)
                self.n_samples = 5000
                with use_seed(42):
                    indices = np.arange(len(dataset))
                    np.random.shuffle(indices)
                    indices = indices[:self.n_samples]
                dataset.data = dataset.data[indices]
                dataset.targets = dataset.targets[indices] if hasattr(dataset, 'targets') else dataset.labels[indices]

            elif split == 'private':
                dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, train=True)

            elif split == 'test':
                dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, train=False)
                self.n_samples = 5000
                with use_seed(42):
                    indices = np.arange(len(dataset))
                    np.random.shuffle(indices)
                    indices = indices[self.n_samples:]
                dataset.data = dataset.data[indices]
                dataset.targets = dataset.targets[indices] if hasattr(dataset, 'targets') else dataset.labels[indices]


        elif self.name == 'usps':
            if split == 'public':
                dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, train=True)
            elif split == 'private':
                dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, train=False)
                self.n_samples = 1007
            if self.n_samples is not None:
                assert self.n_samples < len(dataset)
                with use_seed(42):
                    indices = np.random.choice(range(len(dataset)), self.n_samples, replace=False)
                dataset.data = dataset.data[indices]
                dataset.targets = list(np.array(dataset.targets)[indices])

        self.dataset = dataset

    @property
    def transform(self):
        return Compose([ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label + self.label_shift

class MNISTDataset(_AbstractDataset):
    dataset_class = MNIST
    name = 'mnist'
    n_classes = 10
    n_channels = 1
    img_size = (28, 28)

class USPSDataset(_AbstractDataset):
    dataset_class = USPS
    name = 'usps'
    n_classes = 10
    n_channels = 1
    img_size = (16, 16)
