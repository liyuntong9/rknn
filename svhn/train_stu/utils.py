import logging
import os,sys
import torch
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
import math
from dataset import SVHN_unlabeled, SVHN_labeled
from torchvision.datasets import SVHN
from copy import deepcopy
import numpy as np
import random
from easydict import EasyDict
import yaml



def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)




class Augmentation:
    def __init__(self, K, transform):
        self.transform = transform
        self.K = K

    def __call__(self, x):
        out = [self.transform(x) for _ in range(self.K)]
        return out


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_normalize(_dataset):
    if _dataset =='SVHN':
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    else:
        raise NotImplementedError

def get_mixmatch_transform(_dataset):
    mean, std = get_normalize(_dataset)
    if _dataset == 'SVHN':
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    else:
        raise NotImplementedError

    return train_transform, test_transform


def create_logger(configs):
    result_dir = configs.clu_dir
    out_dir = os.path.join(result_dir, configs.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(result_dir, configs.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    log_file = f'{configs.c}_{configs.k}_{configs.eps_str}_{configs.t}.resume'
    final_log_file = os.path.join(result_dir, log_file)
    print('final: ', final_log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(head))
    logger.addHandler(console_handler)

    if configs.mode =='train':
        logger.info(f"c:{configs.c}, k:{configs.k}, eps:{configs.eps_str}, t:{configs.t}")
        logger.info(f"  Model       = WideResNet {configs.depth}x{configs.width}")
        logger.info(f"  large model = {configs.large}")
        logger.info(f"  Batch size  = {configs.batch_size}")
        logger.info(f"  Epoch       = {configs.epochs}")
        logger.info(f"  Optim       = {configs.optim}")
        logger.info(f"  lambda_u    = {configs.lambda_u}")
        logger.info(f"  alpha       = {configs.alpha}")
        logger.info(f"  T           = {configs.T}")
        logger.info(f"  K           = {configs.K}")
    return logger, writer, out_dir


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def mixmatch_interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def mixmatch_interleave(xy, batch):
    nu = len(xy) - 1
    offsets = mixmatch_interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]






def get_traindata(root, dataset, K, labeled_idx_file, label_file, transform_train=None, transform_val=None):

    if dataset == 'SVHN':

        test_dataset = SVHN(root, split='test', download=False)

        np.random.seed(42)
        rand_indices = np.arange(len(test_dataset))
        np.random.shuffle(rand_indices)

        base_dataset = deepcopy(test_dataset)
        base_dataset.data = base_dataset.data[rand_indices[:-1000]]
        base_dataset.labels = base_dataset.labels[rand_indices[:-1000]]
        val_dataset = deepcopy(test_dataset)
        val_dataset.data = val_dataset.data[rand_indices[-1000:]]
        val_dataset.labels = val_dataset.labels[rand_indices[-1000:]]


        train_labeled_idxs = np.load(labeled_idx_file)
        all_idxs = np.arange(len(base_dataset))
        tmp_flag = np.ones(len(base_dataset)).astype(bool)
        tmp_flag[train_labeled_idxs] = False
        train_unlabeled_idxs = all_idxs[tmp_flag]
        base_dataset.labels[train_labeled_idxs] = np.load(label_file)

        train_labeled_dataset = SVHN_labeled(base_dataset.data, base_dataset.labels, train_labeled_idxs, transform=transform_train)
        train_unlabeled_dataset = SVHN_unlabeled(base_dataset.data, base_dataset.labels, train_unlabeled_idxs, transform=Augmentation(K, transform_train))
        val_dataset = SVHN_labeled(val_dataset.data, val_dataset.labels, transform=transform_val)

        return train_labeled_dataset, train_unlabeled_dataset, val_dataset


def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]


def create_config(yml_file):
    with open(yml_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v
    return cfg

