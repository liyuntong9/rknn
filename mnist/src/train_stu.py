

from torch import optim
import os
from stu_utils import *
from torch.utils.data import DataLoader
from tools import get_logger, create_config, set_seed, to_str
cfg = create_config()
set_seed(cfg.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
data_dir = cfg.data_dir


root_dir = cfg.root_dir

clusters = cfg.clusters
eps_list = cfg.eps_list
neighbors = cfg.neighbors
exp_runs = cfg.exp_runs

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


public_samples = 5000
for c in clusters:
    for k in neighbors:
        for eps in eps_list:
            eps_str = to_str(eps)
            for t in range(exp_runs):
                transform = transforms.Compose([
                    RandomRotation(20, seed=cfg.seed),
                    transforms.RandomAffine(0, translate=(0.2, 0.2)),
                ])
                ass_path = os.path.join(root_dir, f'clu_{c}')
                label_file = os.path.join(ass_path, 'label_results{}_{}_{}_{}.npy'.format(c, eps_str, k, t+1))

                pred_dir = os.path.join(ass_path, 'preds')
                os.makedirs(pred_dir, exist_ok=True)
                log_file = os.path.join(pred_dir, f'log{c}_{k}_{eps_str}_{t + 1}.txt')
                log, log_close = get_logger(log_file)
                log(f'num_cluster: {c}, num_neighbor: {k}, eps: {eps}, num_runs: {t + 1}')
                train_dataset = MnistDataset(data_dir + 'MNIST/raw/', public_samples, is_public=True, transform=transform, label_path=label_file)
                test_dataset = MnistDataset(data_dir + 'MNIST/raw/', public_samples, is_public=False, transform=None)
                train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
                model = ModelM5().to(device)
                ema = EMA(model, decay=0.999)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                criterion = LabelSmoothingCE()
                g_step = 0
                max_correct = 0

                for epoch in range(cfg.epochs):
                    log(f'Epoch: {epoch}')
                    model.train()
                    train_loss = 0
                    train_corr = 0
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target, smoothing=0.45)
                        train_pred = output.argmax(dim=1, keepdim=True)
                        train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
                        train_loss += loss.item() * data.size(0)
                        loss.backward()
                        optimizer.step()
                        g_step += 1
                        ema(model, g_step)

                    train_loss /= len(train_loader.dataset)
                    train_acc = 100 * train_corr / len(train_loader.dataset)

                    log('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \n'.format(
                        train_loss, train_corr, len(train_loader.dataset), train_acc))

                    model.eval()
                    ema.assign(model)
                    test_loss = 0
                    test_corr = 0
                    with torch.no_grad():
                        for data, target in test_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output, target)
                            test_loss += loss.item() * test_loader.batch_size

                            pred = output.argmax(dim=1, keepdim=True)
                            test_corr += pred.eq(target.view_as(pred)).sum().item()
                    if max_correct < test_corr:
                        max_correct = test_corr

                    ema.resume(model)
                    test_loss /= len(test_loader.dataset)
                    test_acc = 100 * test_corr / len(test_loader.dataset)
                    best_test_acc = 100 * max_correct / len(test_loader.dataset)
                    log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) (best: {:.2f}%)\n'.format(
                        test_loss, test_corr, len(test_loader.dataset), test_acc, best_test_acc))
                    lr_scheduler.step()
                log_close()

