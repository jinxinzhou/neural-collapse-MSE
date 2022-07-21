# On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features

This is the code for the [paper](https://proceedings.mlr.press/v162/zhou22c.html) "On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features".

International Conference on Machine Learning (ICML), 2022

## Introduction

- We provide the first global optimization landscape analysis of *Neural Collapse* (NC) under mean-square-error (MSE) loss – an intriguing empirical phenomenon that arises in the last-layer classifiers and features of neural networks during the terminal phase of training. 
- We study the problem based on a simplified *unconstrained feature model*, which isolates the topmost layers from the classifier of the neural network. In this context, we show that the MSE loss with weight decay has a benign global landscape: the only global minimizers are the Simplex Equiangular Tight Frames (ETFs) while all other critical points are strict saddles whose Hessian exhibit negative curvature directions.
- Our experiments demonstrate that one may fix the last-layer classifier to be a Simplex ETF with `d = K` for network training, which reduces memory cost by over 20% on ResNet18 without sacrificing the generalization performance.

## Environment

- CUDA 11.0
- python 3.8.3
- torch 1.6.0
- torchvision 0.7.0
- scipy 1.5.2
- numpy 1.19.1

## Measuring NC during network training

### Datasets

By default, the code assumes the datasets for MNIST and CIFAR10 are stored under `~/data/`. If the datasets are not there, they will be automatically downloaded from `torchvision.datasets`. User may change this default location of datasets in `args.py` through the argument `--data_dir`.

### Training with SGD

~~~python
$ python train_1st_order.py --gpu_id 0 --uid <saving directory name> --dataset <mnist or cifar10> --optimizer SGD --batch_size 256 --lr 0.05
~~~

### Training with Adam

~~~Python
$ python train_1st_order.py --gpu_id 0 --uid <saving directory name> --dataset <mnist or cifar10> --optimizer Adam --batch_size 64 --lr 0.001
~~~

### Training with LBFGS

~~~python
$ python train_2nd_order.py --gpu_id 0 --uid <saving directory name> --dataset <mnist or cifar10> --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048
~~~

**Note:** For each epoch during training, a model will be saved under the directory `model_weights/<the uid name fed to the above commands>/` for the purpose of validating the NC phenomenon in the future. 

There are many other training options, e.g.,   `--epochs`, `--weight_decay` and so on, can be found in `args.py`.

### Validating NC phenomenon

~~~python
$ python validate_NC.py --gpu_id 0 --dataset <mnist or cifar10> --batch_size 256 --load_path <path to the uid name>
~~~

After training, by running the above command, we are able to calculate the four NC metrics defined in the paper. All the information of the NC metrics will be saved in an output file named `info.pkl`. 

Finally, To reproduce the figures in the paper, 1

1.The experiments of different optimization algorithms can be visualized by plotting them in figures:

~~~python
$ python plot_optim_mse.py
~~~

2.The experiment of fixing last-layer classifier can be visualized by plotting them in figures:

~~~python
$ python plot_etf_mse.py
~~~

3.The experiment of changing feature dimension can be visualized by plotting them in figures:

~~~python
$ python plot_dk_mse.py
~~~

4.The experiment of varying rescaled MSE parameters can be visualized by plotting them in figures:

~~~python
$ python plot_M_rmse.py
$ python plot_k_rmse.py
~~~

**Note:** Please refer to one of them for the details of plotting each figure in the paper.

## Validating the unconstrained feature models for NC

### Validity of unconstrained feature models

~~~Python
$ python train_1st_order.py --gpu_id 0 --uid <saving directory name> --dataset cifar10_random --optimizer SGD --batch_size 64 --lr 0.01 --model <MLP or ResNet18_adapt> --width <specify width for model> --depth <specify depth for MLP> --weight_decay 1e-4

$ python validate_NC.py --gpu_id 0 --dataset cifar10_random --batch_size 1000 --load_path <path to the uid name> --model <MLP or ResNet18_adapt> --width <specify width for model> --depth <specify depth for MLP>
~~~

### Weight decay on the network parameter Θ vs. on the features H

~~~Python
$ python train_1st_order.py --gpu_id 0 --uid <saving directory name> --dataset <mnist or cifar10> --optimizer SGD --batch_size 64 --lr 0.05 --model <specify model> --weight_decay <specify weight decay> --sep_decay --feature_decay_rate <specify weight decay on features>
~~~

## Improving network design

### Fix the last-layer classifier as a Simplex ETF

~~~Python
$ python train_1st_order.py --gpu_id 0 --uid <saving directory name> --dataset <mnist or cifar10> --optimizer SGD --lr 0.05 --ETF
~~~

### Feature dimension reduction by choosing `d=K`
~~~Python
$ python train_1st_order.py --gpu_id 0 --uid <saving directory name> --dataset <mnist or cifar10> --optimizer SGD --lr 0.05 --fixdim 10
~~~

### Introduce data augmentation and use modified ResNet architectures
~~~Python
$ python train_1st_order.py --gpu_id 0 --uid <saving directory name> --dataset <mnist or cifar10> --optimizer SGD --lr 0.05 --SOTA
~~~
## Citation and reference 
For technical details and full experimental results, please check [our paper](https://arxiv.org/pdf/2203.01238.pdf).
```
@article{zhou2022optimization,
  title={On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features},
  author={Zhou, Jinxin and Li, Xiao and Ding, Tianyu and You, Chong and Qu, Qing and Zhu, Zhihui},
  journal={arXiv preprint arXiv:2203.01238},
  year={2022}
}
```
