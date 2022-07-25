##This subfolder contains the implementation code of RKNN on MNIST dataset.
First, we use DTI clustering to divide public data into 10-100 cluters.
Then, we take cluster centers as query samples and connect private records to k-nearest queries.
Afer voting, we add noise to label vectors for the purpose of centralized/local DP. 
Finally, the pseudo-labeled public samples are employed to train student model.

In DPSGD, we obtain a public representation model with the aid of DTI and extract private data into latent features.
With these private training features, we train a simple two-layer perceptron as classification network.

###&#x1F680; Installation
This code depends on  DTI clustering, and therefore you can refer to [DTI repo](https://github.com/monniert/dti-clustering) for installation.
Also you can simplely run:
```
conda env create -f rknn_mnist.yml
conda activate rknn_mnist
```
###&#x1F308; Launch training 
####1. cluster public data
```
cd clustering 
### modify configs/mnist.yml to set the number of clusters
python train.py --tag exp_name --config mnist.yml
```
####2. vote for public data
```
### modify rknn_configs.yml to set hyperparams
python vote.py
python vote_ldp.py
```
####3. train student model
 ```
python train_stu.py
python train_stu_ldp.py
```
