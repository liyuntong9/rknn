import math
import numpy as np
import torch
import torchvision.transforms as transforms
from .collate import collate_custom
import os
 
def get_criterion(p):
    if p['criterion'] == 'simclr':
        from .losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])
    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))
    return criterion


def get_model(p, pretrain_path=None):
    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in ['cifar-10',]:
            from .resnet_cifar import resnet18
            backbone = resnet18()
        else:
            raise NotImplementedError
    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    if p['setup'] in ['simclr',]:
        from .models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])
    elif p['setup'] in ['scan', 'selflabel']:
        from .models import ClusteringModel
        if p['setup'] == 'selflabel':
            assert(p['num_heads'] == 1)
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])
    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))



    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')

        if p['setup'] == 'scan':
            missing = model.load_state_dict(state, strict=False)
            assert (set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias',
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                    or set(missing[1]) == {
                        'contrastive_head.weight', 'contrastive_head.bias'})

        elif p['setup'] == 'selflabel':
            model_state = state['model']
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['cluster_head.%d.weight' % (state['head'])]
            best_head_bias = model_state['cluster_head.%d.bias' % (state['head'])]
            for k in all_heads:
                model_state.pop(k)

            model_state['cluster_head.0.weight'] = best_head_weight
            model_state['cluster_head.0.bias'] = best_head_bias
            missing = model.load_state_dict(model_state, strict=True)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model


def get_train_dataset(p, transform, to_augmented_dataset=False):
    if p['train_db_name'] == 'cifar-10':
        from .cifar import CIFAR10
        dataset = CIFAR10(root=p['data_dir'], train=True, transform=transform, download=True)
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    if to_augmented_dataset:
        from .custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)
    
    return dataset


def get_val_dataset(p, transform=None):
    if p['val_db_name'] == 'cifar-10':
        from .cifar import CIFAR10
        dataset = CIFAR10(root=p['data_dir'], train=False, transform=transform, download=True)
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    return dataset


def get_test_dataset(p, transform=None):
    if p['val_db_name'] == 'cifar-10':
        from .cifar import CIFAR10
        dataset = CIFAR10(root=p['data_dir'], train=False, transform=transform, download=True, knowledge_test=True)
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    return dataset

def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):

    if p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    return transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(), 
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])


def get_optimizer(p, model):

    params = model.parameters()
                

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
