This subfolder contains the implementation code of RKNN on SVHN dataset.
First, we use HOG & K-means++ to extract public features and cluster. 
Then, we take cluster centers as query samples and connect private records to k-nearest queries.
Afer voting, we add noise to label vectors for the purpose of centralized/local DP. 
Finally, the pseudo-labeled queries are employed for the semi-supervised training of student model.

In DPSGD, we use HOG to extract private data into latent features.
With these private training features, we train a simple two-layer perceptron as classification network.

### &#x1F308; Installation

This code depends on semi-supervised training algorithm Mixmatch, and therefore you can refer to [Mixmatch repo](https://github.com/Jeffkang-94/Mixmatch-pytorch-SSL) for installation.
Also you can simplely run:
```
conda env create -f rknn_svhn.yml
conda activate rknn_svhn
```

### &#x1F680; Launch a training

#### 1. get public features
```
# modify env.yml to set hyperparams
python hog.py 
```
#### 2. cluster public data
```
python cluster.py
```
#### 3. vote for queries
```
python vote.py
```
#### 4. train student model
 ```
cd train_stu
python main.py
```
