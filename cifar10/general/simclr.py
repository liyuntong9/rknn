import os
import torch
import numpy as np
from utils.config import create_config

from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate, get_test_dataset


from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

import  os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main():
    config_env = 'utils/env.yml'
    config_exp = 'utils/simclr_cifar10.yml'
    p = create_config(config_env, config_exp)
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    model = model.cuda()
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    train_transforms = get_train_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)

    val_transforms = get_val_transformations(p)
    val_dataset = get_val_dataset(p, val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_val_dataloader(p, test_dataset)
    print('Dataset contains {}/{}/{} train/val/test samples'.format(len(train_dataset), len(val_dataset), len(test_dataset)))


    base_dataset = get_train_dataset(p, val_transforms)
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()


    memory_bank_test = MemoryBank(len(test_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_test.cuda()

    criterion = get_criterion(p)
    criterion = criterion.cuda()
    optimizer = get_optimizer(p, model)

    print('checkpoint path: ', p['simclr_checkpoint'])
    if os.path.exists(p['simclr_checkpoint']):

        print(colored('Restart from checkpoint {}'.format(p['simclr_checkpoint']), 'blue'))
        checkpoint = torch.load(p['simclr_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        start_epoch = 0
        model = model.cuda()

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)


        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['simclr_checkpoint'])

    fill_memory_bank(base_dataloader, model, memory_bank_base)

    fill_memory_bank(val_dataloader, model, memory_bank_val)
    fill_memory_bank(test_dataloader, model, memory_bank_test)
    np.save(os.path.join(p['feats_dir'], 'public_feats.npy'), memory_bank_base.features.detach().cpu().numpy())
    np.save(os.path.join(p['feats_dir'], 'public_labels.npy'), memory_bank_base.targets.detach().cpu().numpy())
    np.save(os.path.join(p['feats_dir'], 'private_feats.npy'), memory_bank_val.features.detach().cpu().numpy())
    np.save(os.path.join(p['feats_dir'], 'private_labels.npy'), memory_bank_val.targets.detach().cpu().numpy())

    np.save(os.path.join(p['feats_dir'], 'test_feats.npy'), memory_bank_test.features.detach().cpu().numpy())
    np.save(os.path.join(p['feats_dir'], 'test_labels.npy'), memory_bank_test.targets.detach().cpu().numpy())




 
if __name__ == '__main__':
    main()
