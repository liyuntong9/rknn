from backpack import backpack, extend
from backpack.extensions import BatchGrad
from rdp_accountant import get_sigma, adjust_learning_rate, process_grad_batch
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from model import MLP, get_logger
import time
import os
os.environ["VISIBLE_CUDA_DEVICES"]="1"
clip = 5.0
dims = [64, 32]
lr= 0.1
momentum=0.9
n_epoch =1
batchsize = 128

eps = 1.0
delta = 1. / 60000
seed = 42

log, log_close =get_logger('mnist_eps1.txt')
feats_dir = '/kolla/code/lyt/general_rknn/mnist/src/clustering/runs/mnist/clu_10'
private_feats = np.load(os.path.join(feats_dir, 'private_feats.npy'))
private_labels = np.load(os.path.join(feats_dir, 'private_labels.npy'))
test_feats = np.load(os.path.join(feats_dir, 'test_feats.npy'))
test_labels = np.load(os.path.join(feats_dir, 'test_labels.npy'))
x_train = torch.from_numpy(private_feats).float()
y_train = torch.from_numpy(private_labels).long()
x_test = torch.from_numpy(test_feats).float()
y_test = torch.from_numpy(test_labels).long()
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True, drop_last=True)
test_ds = TensorDataset(x_test, y_test)
test_dl = DataLoader(test_ds, batch_size=100, shuffle=False)
loader = iter(train_dl)
n_training = int(len(train_ds) / batchsize) * batchsize

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
sampling_prob = batchsize/n_training
steps = int(n_epoch * n_training / batchsize)
sigma, eps = get_sigma(sampling_prob, steps, eps, delta, rgp=False)
noise_multiplier = sigma
best_acc = 0
net = MLP(dims).cuda()
net = extend(net)

num_params = 0
for p in net.parameters():
    num_params += p.numel()

log(f'total number of parameters: {num_params/(10**6)} M')

loss_func = nn.CrossEntropyLoss(reduction='sum')
loss_func = extend(loss_func)


optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum)


def train(epoch):
    log('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_dl):

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        with backpack(BatchGrad()):
            loss.backward()
            process_grad_batch(list(net.parameters()), clip)
            for p in net.parameters():
                grad_noise = torch.normal(0, noise_multiplier*clip/batchsize, size=p.grad.shape, device=p.grad.device)
                p.grad.data += grad_noise

        optimizer.step()
        step_loss = loss.item()
        step_loss /= inputs.shape[0]
        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
    acc = 100.*float(correct)/float(total)
    t1 = time.time()
    log('Train loss:%.5f'%(train_loss/(batch_idx+1)) +  'time: %d s'%(t1-t0) + 'train acc:%.5f' % acc)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            step_loss = loss.item()
            step_loss /= inputs.shape[0]

            test_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().item()

        acc = 100.*float(correct)/float(total)
        log('test loss:%.5f'%(test_loss/(batch_idx+1)) + 'test acc:%.5f'%acc)
        if acc > best_acc:
            best_acc = acc
            log('best test acc:%.5f'% best_acc)

for epoch in range(0, n_epoch):
    lr = adjust_learning_rate(optimizer, lr, epoch, all_epoch=n_epoch)
    train(epoch)
    test()

log_close()
