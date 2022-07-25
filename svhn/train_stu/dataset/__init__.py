from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class SSL_Dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _transpose(self, x, source='NHWC', target='NCHW'):
        return x.transpose([source.index(d) for d in target])

    def _get_PIL(self, x):
        return Image.fromarray(x)


class SVHN_labeled(SSL_Dataset):

    def __init__(self, data, targets, indexes=None, transform=None, target_transform=None):
        super(SVHN_labeled, self).__init__(transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = targets
        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.transpose(img, (1,2,0)))
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                img = self.transform[0](img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        leng = len(self.data)
        return leng


class SVHN_unlabeled(SVHN_labeled):
    def __init__(self, data, targets, indexes=None,
                 transform=None, target_transform=None):
        super(SVHN_unlabeled, self).__init__(data, targets, indexes, transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for _ in range(len(self.targets))])








