# Semantic Graph Convolutional Networks

This repository is the official implementation of Semantic Graph Convolutional Networks.

The ML Code Completness Checklist consists of five items:

1. **Specification of dependencies**
2. **Code**
3. **Data**

We explain each item on the checklist in detail blow. 

#### 1. Specification of dependencies

If you are using Python, this means providing a `requirements.txt` file.

You can choose 'pip' or 'conda' command to install dependencies in `requirements.txt`. We suggest you create a new virtual environment to try our code.

In our experiments, we trained our model via 8 32G NVIDIA Tesla V100 GPUs.

#### 2. Code

At present, we provide the code file `SGCN.py` which includes the training module and the evaluation module. We will sort out our code more systematically in future.

Due to the capacity limitations, we give the specific parameter(via hyperopt optimization) of runing process on Cora, Citeseer and Pubmed in major experiment 'Semi-Supervised Node Classification'. For the other details such as hyperopt file, we would release all of them if the paper was luckily to be accepted.

It is easily to train and eval the model as:

1) Cora
````
CUDA_VISIBLE_DEVICES=0 python SGCN.py --datname cora --nbsz 20 --alpha 10 --dropout 0.058 --lr 0.045 --ncaps 7 --nhidden 10 --nlayer 8 --reg 0.002 --routit 6
````

2) Pubmed
````
CUDA_VISIBLE_DEVICES=2 python SGCN.py --datname pubmed  --nbsz 20 --alpha 1.5 --dropout 0.27 --lr 0.02 --ncaps 7 --nhidden 24 --nlayer 8 --reg 0.022 --routit 6
````

3) Citeseer

The highest accuracy on Citeseer is given by SGCN-meta.py.
````
CUDA_VISIBLE_DEVICES=2 python SGCN-meta.py --datname citeseer --nbsz 20 --dropout 0.18 --lr 0.016 --ncaps 7 --nhidden 20 --nlayer 8 --reg 0.058 --routit 6
````

#### 3. Data

The dataset of Cora, Citeseer and Pubmed are given in /notebooks/data/. We show the details in Table 1.

![avatar](/dataset_info.png)

I hope our document can help you！
