
from torch import nn
import torch.nn.functional as F
import random
import torchvision.transforms.functional as tF
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class ModelM5(nn.Module):
    def __init__(self):
        super(ModelM5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(10240, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def forward(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        flat5 = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat5))
        return logits



class LabelSmoothingCE(nn.Module):
    def __init__(self):
        super(LabelSmoothingCE, self).__init__()
    def forward(self, logits, targets, smoothing=0.1):
        confidence = 1 - smoothing
        tmp = -F.log_softmax(logits, dim=-1)
        nll_loss = tmp.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = tmp.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return  loss.mean()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class RandomRotation(object):
    def __init__(self, degrees, seed=42):
        self.degrees = (-degrees, degrees)
        random.seed(seed)

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        return tF.rotate(img, angle, False, False, None, None)




class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, dset_root, public_samples, is_public=True, transform=None, label_path=None):

        f = open(dset_root + 't10k-images-idx3-ubyte', 'rb')
        xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
        f.close()
        f = open(dset_root + 't10k-labels-idx1-ubyte', 'rb')
        ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
        f.close()
        xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
        ys = ys.astype(np.int)
        rand_indices = np.arange(xs.shape[0])
        np.random.shuffle(rand_indices)
        self.x_data = xs
        self.y_data = ys
        if is_public:
            self.x_data = xs[rand_indices[:public_samples]]
            label_results = np.load(label_path)
            self.y_data = label_results
        else:
            self.x_data = xs[rand_indices[public_samples:]]
            self.y_data = ys[rand_indices[public_samples:]]
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_data[idx].reshape(28, 28))
        y = torch.tensor(np.array(self.y_data[idx]))
        if self.transform:
            x = self.transform(x)
        x = transforms.ToTensor()(np.array(x) / 255)
        return x, y
