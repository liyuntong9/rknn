##This folder contains the implementation code for RKNN on Cifar10 dataset.
First, we use Simclr & K-means++ to train a public representation model and cluster public features (denoted as general).
Likewise, we can employ SCAN clustering to divide public data into 10 cluters (denoted as end2end).
Then, we regard cluster centers as query samples and connect private records to k-nearest queries.
Afer voting and aggregating, we add noise to label vectors for the purpose of centralized/local DP. 
Finally, these pseudo-labeled public samples are employed to train student model.

In DPSGD, we obtain a public representation model with the aid of Simclr and extract private data into latent features.
With these private training features, we train a simple two-layer perceptron as classification network.

### &#x1F308; Installation
This code depends on SCAN clustering, and therefore you can refer to [SCAN repo](https://github.com/wvangansbeke/Unsupervised-Classification) for installation.
Also you can simplely run:
```
conda env create -f rknn_cifar10.yml
conda activate rknn_cifar10
```
### &#x1F680; Launch training (general)
####1. train public representation model
```
python simclr.py 
```
####2. cluster public data
```
### modify rknn_configs.yml to set hyperparams
python cluster.py
```
####3. vote for queries
```
python vote.py
```
####4. train student model
 ```
python train_stu.py
```

### &#x1F680; Launch training (end2end)
####1. cluster public data
```
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
python scan.py --config_env configs/env.yml  --config_exp configs/scan/scan_cifar10.yml
python selflabel.py --config_env configs/env.yml   --config_exp configs/selflabel/selflabel_cifar10.yml
python gen_feats.py 
```
####2. vote for queries
```
python vote.py
```
####3. train student model
 ```
python train_stu.py
```


This repository contains the implementation code of Record-level Private Knowledge Distillation [[arxiv]()].
Enter subfolders to view the guide for running.
![Framework]()

