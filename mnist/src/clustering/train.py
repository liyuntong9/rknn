import argparse
from datasets import get_dataset
from optimizer import get_optimizer
from scheduler import get_scheduler
from models import get_model
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from utils import use_seed, coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
import yaml
from utils.path import CONFIGS_PATH, RUNS_PATH
import shutil
from utils.logger import get_logger, print_info
from utils.metrics import Metrics, Scores, AverageMeter
from models.tools import safe_model_state_dict
PRINT_LR_UPD_FMT = "Epoch [{}/{}], Iter [{}/{}], LR Update: {}".format
PRINT_TRAIN_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], Train Metrics: {}".format
PRINT_CHECK_CLUSTERS_FMT = "Epoch [{}/{}], Iter [{}/{}], Reassigned clusters {} from cluster {}".format
MODEL_FILE = 'model.pkl'
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class Trainer:
    @use_seed()
    def __init__(self, config, run_dir):
        self.config = coerce_to_path_and_check_exist(config)
        self.run_dir = coerce_to_path_and_create_dir(run_dir)
        self.logger = get_logger(self.run_dir, name='trainer')
        shutil.copy(self.config, self.run_dir)
        with open(self.config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        # dataset
        self.dset_kwargs = cfg['dataset']
        self.dset_name = self.dset_kwargs.pop('name')
        train_ds = get_dataset(self.dset_name)('public')
        self.n_classes = train_ds.n_classes
        self.batch_size = cfg['training']['batch_size']
        self.n_workers = cfg['training'].get('n_workers', 4)
        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
        self.n_batches = len(self.train_dl)
        self.n_iters = cfg['training'].get('n_iters')
        self.n_epochs = cfg['training'].get('n_epochs')
        self.knn = cfg['training'].get('knn')
        assert (self.n_iters is not None or self.n_epochs is not None)
        if self.n_iters is not None:
            self.n_epochs = max(self.n_iters // self.n_batches, 1)
        else:
            self.n_iters = self.n_epochs * len(self.train_dl)

        # model
        self.model_kwargs = cfg['model']
        self.model_name = self.model_kwargs.pop('name')
        self.is_gmm = 'gmm' in self.model_name
        self.model = get_model(self.model_name)(self.train_dl.dataset, **self.model_kwargs).to(self.device)
        self.n_prototypes = self.model.n_prototypes


        # optimizer
        opt_params = cfg['training']['optimizer']
        opt_name = opt_params.pop('name')
        cluster_kwargs = opt_params.pop('cluster', {})
        tsf_kwargs = opt_params.pop('transformer', {})
        self.optimizer = get_optimizer(opt_name)([dict(params=self.model.cluster_parameters(), **cluster_kwargs),
                                                  dict(params=self.model.transformer_parameters(), **tsf_kwargs)], **opt_params)
        self.model.set_optimizer(self.optimizer)
        scheduler_params = cfg['training'].get('scheduler', {})
        scheduler_name = scheduler_params.pop('name', None)
        self.schd_update = scheduler_params.pop('update_range', 'epoch')
        assert self.schd_update in ['epoch', 'batch']
        if scheduler_name == 'multi_step' and isinstance(scheduler_params['milestones'][0], float):
            n_tot = self.n_epochs if self.schd_update == 'epoch' else self.n_iters
            scheduler_params['milestones'] = [round(m * n_tot) for m in scheduler_params['milestones']]
        self.scheduler = get_scheduler(scheduler_name)(self.optimizer, **scheduler_params)
        self.cur_lr = self.scheduler.get_last_lr()[0]

        # resume
        checkpoint_resume = cfg['training'].get('resume')
        if checkpoint_resume is not None:
            self.load_from_tag(checkpoint_resume)
        else:
            self.start_epoch, self.start_batch = 1, 1

        metric_names = ['time/img', 'loss']
        metric_names += [f'prop_clus{i}' for i in range(self.n_prototypes)]
        self.train_stat_interval = cfg['training']['train_stat_interval']
        self.train_metrics = Metrics(*metric_names)
        self.check_cluster_interval = cfg['training']['check_cluster_interval']

    @use_seed()
    def run(self):
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_check_cluster_iter = cur_iter, cur_iter
        if self.start_epoch == self.n_epochs:
            self.print_and_log("No training, only evaluating")
            self.evaluate('public')
            self.evaluate('private')
            return None
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            start_batch = self.start_batch if epoch == self.start_epoch else 1
            for batch, (images, labels) in enumerate(self.train_dl, start=1):
                if batch < start_batch:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iters:
                    break
                self.single_train(images)

                if self.schd_update == 'batch':
                    self.update_scheduler(epoch, batch)

                if (cur_iter - prev_train_stat_iter) >= self.train_stat_interval:
                    prev_train_stat_iter = cur_iter
                    self.log_train_metrics(epoch, batch)
                if (cur_iter - prev_check_cluster_iter) >= self.check_cluster_interval:
                    prev_check_cluster_iter = cur_iter
                    self.check_cluster(epoch, batch)
                    self.save(epoch=epoch, batch=batch)

            self.model.step()
            if self.schd_update == 'epoch' and start_batch == 1:
                self.update_scheduler(epoch + 1, batch=1)
        self.evaluate('public')
        self.evaluate('private')
        self.print_and_log("Training is over")
    def log_train_metrics(self, epoch, batch):
        self.print_and_log(PRINT_TRAIN_STAT_FMT(epoch, self.n_epochs, batch, self.n_batches, self.train_metrics))
        self.train_metrics.reset('time/img', 'loss')

    def update_scheduler(self, epoch, batch):
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            self.print_and_log(PRINT_LR_UPD_FMT(epoch, self.n_epochs, batch, self.n_batches, lr))

    def print_and_log(self, string):
        print_info(string)
        self.logger.info(string)

    def single_train(self, images):
        start_time = time.time()
        B = images.size(0)
        self.model.train()
        images = images.to(self.device)
        self.optimizer.zero_grad()
        loss, distances = self.model(images)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            min_idx = distances.min(1)[1]
            mask = torch.zeros(B, self.n_prototypes, device=self.device).scatter(1, min_idx[:,None], 1)
            proportions = mask.sum(0).cpu().numpy() / B

        self.train_metrics.update({
            'time/img': (time.time() - start_time) / B,
            'loss': loss.item()
        })
        self.train_metrics.update({f'prop_clus{i}': p for i, p in enumerate(proportions)})

    def save(self, epoch, batch):
        state = {
            'epoch': epoch,
            'batch': batch,
            'model_name': self.model_name,
            'model_kwargs': self.model_kwargs,
            'model_state': self.model.state_dict(),
            'n_prototypes': self.n_prototypes,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.print_and_log('models saved at {}'.format(save_path))
    def check_cluster(self, epoch, batch):
        proportions = [self.train_metrics[f'prop_clus{i}'].avg for i in range(self.n_prototypes)]
        reassigned, idx = self.model.reassign_empty_clusters(proportions)
        self.print_and_log(PRINT_CHECK_CLUSTERS_FMT(epoch, self.n_epochs, batch, self.n_batches, reassigned, idx))
        self.train_metrics.reset(*[f'prop_clus{i}' for i in range(self.n_prototypes)])


    def load_from_tag(self, tag):
        self.print_and_log('loading models!')
        path = coerce_to_path_and_check_exist(RUNS_PATH / self.dset_name / tag / MODEL_FILE)
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except RuntimeError:
            state = safe_model_state_dict(checkpoint['model_state'])
            self.model.module.load_state_dict(state)
        self.start_epoch, self.start_batch = checkpoint['epoch'], checkpoint.get('batch', 0) + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.print_and_log('Checkpoint loaded at epoch {}, batch {}'.format(self.start_epoch, self.start_batch-1))

    def evaluate(self, split):
        self.print_and_log("Start evaluation")
        self.model.eval()
        self.quantitative_eval(split)
        self.get_test_feats()
        self.print_and_log("Evaluation is over")

    @torch.no_grad()
    def get_test_feats(self):
        dataset = get_dataset(self.dset_name)('test')
        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)

        np.save(self.run_dir / 'test_labels.npy', dataset.dataset.targets.numpy())

        self.model.reset_public_feats()
        for idx, (images, labels) in enumerate(data_loader):
            images = images.to(self.device)
            distances = self.model(images, 'public')[1]


        np.save(self.run_dir / 'test_feats.npy', self.model.public_feats)


    @torch.no_grad()
    def quantitative_eval(self, split):
        loss = AverageMeter()
        scores = Scores(self.n_classes, self.n_prototypes)
        dataset = get_dataset(self.dset_name)(split)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)

        if split == 'public':
            ass = np.zeros(len(dataset), dtype=np.int64)
            np.save(self.run_dir / 'public_labels.npy', dataset.dataset.targets.numpy())
        else:
            ass_dict = {k: np.zeros((len(dataset),k), dtype=np.int64) for k in range(1, self.knn + 1)}
            np.save(self.run_dir / 'private_labels.npy', dataset.dataset.targets.numpy())
        self.model.reset_public_feats()
        self.model.reset_private_feats()
        for idx, (images, labels) in enumerate(data_loader):
            images = images.to(self.device)
            distances = self.model(images, split)[1]
            min_dist, min_idx = distances.min(1)
            if split == 'public':
                ass[idx * self.batch_size: (idx + 1) * self.batch_size] = min_idx.cpu().numpy()
            else:
                for k in range(1, self.knn+1):
                    neighbors = torch.topk(distances, k, largest=False)[1].cpu().numpy()
                    ass_dict[k][idx * self.batch_size: (idx + 1) * self.batch_size] = neighbors

            loss.update(min_dist.mean(), n=len(min_dist))
            scores.update(labels.long().numpy(), min_idx.cpu().numpy())
        if split == 'public':
            np.save(self.run_dir / (split + '_ass{}.npy'.format(self.n_prototypes)), ass)
        else:
            for k in range(1, self.knn + 1):
                np.save(self.run_dir / (split + '_ass{}_{}.npy'.format(self.n_prototypes, k)), ass_dict[k])
        scores = scores.compute()
        np.save(self.run_dir / 'public_feats.npy', self.model.public_feats)
        np.save(self.run_dir / 'private_feats.npy', self.model.private_feats)
        self.print_and_log("final_loss: {:.5}".format(loss.avg))
        self.print_and_log("final_scores: " + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()]))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Transformation Invariant Clustering")
    parser.add_argument("-t", "--tag", nargs="?", type=str, required=True, help="Run tag of the experiment")
    parser.add_argument("-c", "--config", nargs="?", type=str, required=True, help="Config file name")
    args = parser.parse_args()
    assert args.tag is not None and args.config is not None
    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    seed = cfg['training'].get('seed', 42)
    dataset = cfg['dataset'].get('name')
    run_dir = RUNS_PATH / dataset / args.tag
    trainer = Trainer(config, run_dir, seed=seed)
    trainer.run(seed=seed)

