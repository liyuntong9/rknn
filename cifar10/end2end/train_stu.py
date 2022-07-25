

from data.cifar import CIFAR10
from stu_model import DenseNet121

import torch
import os
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from stu_tools import create_config, set_seed, get_logger


cfg = create_config()
set_seed(cfg.seed)
cluster = cfg.cluster
neighbors = cfg.neighbors
eps_list = cfg.eps_list
exp_runs = cfg.exp_runs
feats_dir = cfg.feats_dir
dp_type = cfg.dp_type
data_dir = cfg.data_dir
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

epochs = cfg.epochs

for k in neighbors:
    for eps in eps_list:
        for t in range(exp_runs):

            if str(eps).find('.') != -1:
                list_str = list(str(eps))
                list_str.pop(str(eps).find('.'))
                eps_str = ''.join(list_str)
            else:
                eps_str = str(eps)

            type_dir = os.path.join(feats_dir, dp_type)
            pred_dir = os.path.join(type_dir, 'pred_dir')
            label_results = np.load(os.path.join(type_dir, f'label_results{cluster}_{k}_{eps_str}_{t+1}.npy'))
            assert (label_results.shape[0]) == 29000
            os.makedirs(pred_dir, exist_ok=True)

            log_file = os.path.join(pred_dir, f'log{cluster}_{k}_{eps_str}_{t+1}.txt')
            log, log_close = get_logger(log_file)

            log(f'num_cluster: {cluster}, num_neighbor: {k}, eps: {eps}, num_runs: {t}')
            device ='cuda' if torch.cuda.is_available() else 'cpu'


            train_set = CIFAR10(root=cfg['data_dir'], train=True, download=True, transform=train_tf, knowledge_test=True,label_results=label_results)
            train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

            test_set = CIFAR10(root=cfg['data_dir'], train=False, download=True, transform=test_tf, knowledge_test=True)
            test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
            log('train samples: %d' % train_set.data.shape[0])
            log('test samples: %d' % test_set.data.shape[0])
            net = DenseNet121()
            net = net.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            def train():
                net.train()
                train_loss = 0
                correct = 0
                for batch_idx, data in enumerate(train_loader):
                    inputs, targets = data['image'], data['target']
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                    predicted = outputs.max(1)[1]
                    correct += predicted.eq(targets).sum().item()

                log('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / len(train_set), 100. * correct / len(train_set), correct, len(train_set)))


            def test(epoch, best_acc):
                net.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for batch_idx, data in enumerate(test_loader):
                        inputs, targets = data['image'], data['target']
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        test_loss += loss.item() * inputs.size(0)
                        predicted = outputs.max(1)[1]
                        correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / len(test_set)
                log('Test Acc: %.3f' % acc)
                if acc > best_acc[0]:
                    state = {
                        'net': net.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    file = os.path.join(pred_dir, f'ckpt{cluster}_{k}_{eps_str}_{t+1}.pth')
                    torch.save(state, file)
                    best_acc[0] = acc
                    log('Best Acc: %.4f' % best_acc[0])
            best_acc = [0]
            for epoch in range(epochs):
                log('Epoch: %d' % epoch)
                start_time = time.time()
                train()
                test(epoch, best_acc)
                scheduler.step()
                end_time = time.time()
                time_elapsed = end_time - start_time
                log('run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            log('finish training \n\n')
            log_close()
