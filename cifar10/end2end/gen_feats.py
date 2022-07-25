
import torch
import yaml
from termcolor import colored
from utils.common_config import get_val_transformations, get_val_dataloader,\
                                get_model, get_train_dataset, get_val_dataset
from utils.evaluate_utils import get_predictions
import numpy as np
import os
from stu_tools import create_config

config_exp = '/kolla/code/lyt/general_rknn/cifar10/end2end/configs/selflabel/selflabel_cifar10.yml'
model_dir = '/kolla/code/lyt/general_rknn/cifar10/end2end/cifar-10/selflabel/model.pth.tar'
cfg = create_config()
cluster = 10
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
feats_dir = '/kolla/code/lyt/general_rknn/cifar10/end2end/feats'
os.makedirs(feats_dir, exist_ok=True)
def main():

    with open(config_exp, 'rb') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512

    transforms = get_val_transformations(config)
    public_dataset = get_train_dataset(config, transforms)
    public_dataloader = get_val_dataloader(config, public_dataset)

    private_dataset = get_val_dataset(config, transforms)
    private_dataloader = get_val_dataloader(config, private_dataset)

    print('Number of public samples: {} / private samples: {}'.format(len(public_dataset), len(private_dataset)))


    print(colored('Get model ...', 'blue'))
    model = get_model(config)
    print(model)
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(model_dir, map_location='cpu')
    if config['setup'] in ['selflabel']:
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError

    model.cuda()


    if config['setup'] in ['scan', 'selflabel']:
        print(colored('Perform evaluation of the clustering model (setup={}).'.format(config['setup']), 'blue'))
        public_preds, public_feats = get_predictions(config, public_dataloader, model, return_features=True)
        private_preds, private_feats = get_predictions(config, private_dataloader, model, return_features=True)
        public_labels = public_preds[0]['targets'].detach().numpy()
        label_pred = public_preds[0]['predictions'].detach().numpy()

        np.save(os.path.join(feats_dir, 'private_feats.npy'), private_feats.detach().numpy())
        np.save(os.path.join(feats_dir, 'private_labels.npy'), private_preds[0]['targets'].detach().numpy())
        np.save(os.path.join(feats_dir, 'public_feats.npy'), public_feats.detach().numpy())
        np.save(os.path.join(feats_dir, 'public_labels.npy'), public_labels)
        np.save(os.path.join(feats_dir, 'label_pred.npy'), label_pred)

        centroids = []
        train_idxs = torch.arange(public_feats.size(0))
        tmp_pred = public_preds[0]['predictions']
        for c in range(cluster):
            mask = tmp_pred == c
            masked_feats = public_feats[mask]
            cen_dis = (masked_feats - masked_feats.mean(dim=0))
            cen_dis = torch.norm(cen_dis, p=2, dim=1)
            cen_idx = cen_dis.min(dim=0)[1]
            centroids.append(train_idxs[mask][cen_idx].item())
        np.save(os.path.join(feats_dir, 'centroids.npy'), np.array(centroids))

    else:
        raise NotImplementedError



if __name__ == "__main__":
    main()
