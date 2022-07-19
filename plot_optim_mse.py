import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

datasets = ['mnist', 'cifar10']
print(os.getcwd())

PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/')

PATH_TO_INFO_SGD = os.path.join(PATH_TO_INFO, 'cifar10_MSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/' + 'info.pkl')
PATH_TO_INFO_ADAM = os.path.join(PATH_TO_INFO, 'cifar10_MSE_Adam_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/'+ 'info.pkl')
PATH_TO_INFO_LBFGS = os.path.join(PATH_TO_INFO, 'cifar10_MSE_LBFGS_bias_true_batchsize_512_ETFfc_false_fixdim_false_sota_false/'+ 'info.pkl')

with open(PATH_TO_INFO_SGD, 'rb') as f:
    info_sgd = pickle.load(f)

with open(PATH_TO_INFO_ADAM, 'rb') as f:
    info_adam = pickle.load(f)

with open(PATH_TO_INFO_LBFGS, 'rb') as f:
    info_lbfgs = pickle.load(f)
#
#
out_path = os.path.join(os.path.dirname(PATH_TO_INFO), '/imgs/mse/optim/')
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)
#
with open(PATH_TO_INFO_SGD, 'rb') as f:
    info_sgd = pickle.load(f)

with open(PATH_TO_INFO_ADAM, 'rb') as f:
    info_adam = pickle.load(f)

with open(PATH_TO_INFO_LBFGS, 'rb') as f:
    info_lbfgs = pickle.load(f)


XTICKS = [0, 50, 100, 150, 200]


def plot_collapse():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['collapse_metric'], 'r', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['collapse_metric'], 'b',  marker='o', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['collapse_metric'], 'g',  marker='s', ms=16,  markevery=25, linewidth=5, alpha=0.7)


    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-2, 6.01, 2), fontsize=30)

    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=1)

    plt.axis([0, 200, -0.2, 6])
    fig.savefig(out_path + "resnet18-NC1.pdf", bbox_inches='tight')


def plot_ETF():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['ETF_metric'], 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['ETF_metric'], 'b',  marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['ETF_metric'], 'g',  marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)


    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 0.81, .2), fontsize=30)

    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=1)

    plt.axis([0, 200, -0.02, 0.8])

    fig.savefig(out_path + "resnet18-NC2.pdf", bbox_inches='tight')


def plot_WH_relation():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['WH_relation_metric'], 'r', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['WH_relation_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['WH_relation_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)


    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=30)


    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=1)

    plt.axis([0, 200, 0, 1])

    fig.savefig(out_path + "resnet18-NC3.pdf", bbox_inches='tight')


def plot_residual():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['Wh_b_relation_metric'], 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['Wh_b_relation_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['Wh_b_relation_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 0.51, 0.1), fontsize=30)

    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=1)
    plt.axis([0, 200, 0, 0.5])

    fig.savefig(out_path + "resnet18-NC4.pdf", bbox_inches='tight')


def plot_train_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['train_acc1'], 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['train_acc1'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['train_acc1'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)


    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(20, 110, 20), fontsize=30)

    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4)

    plt.axis([0, 200, 20, 102])

    fig.savefig(out_path + "resnet18-train-acc.pdf", bbox_inches='tight')


def plot_test_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['test_acc1'], 'r', marker='v',  ms=16,   markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['test_acc1'], 'b', marker='o',  ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['test_acc1'], 'g', marker='s', ms=16,  markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(30, 80.1, 10), fontsize=30)

    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4)

    plt.axis([0, 200, 30, 80])

    fig.savefig(out_path + "resnet18-test-acc.pdf", bbox_inches='tight')


def plot_cos_margin_distribution():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    ITICKS = [0, 10000, 20000, 30000, 40000, 50000]

    plt.plot(info_sgd['all_cos_margin'][-1], 'r', marker='v', ms=16, markevery=10000, linewidth=5, alpha=0.7)
    plt.plot(info_adam['all_cos_margin'][-1], 'b', marker='o', ms=16, markevery=10000, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['all_cos_margin'][-1], 'g', marker='s', ms=16, markevery=10000, linewidth=5, alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel(r'$\mathcal{P}_{CM}$', fontsize=40)
    plt.xticks(ITICKS, fontsize=30)

    plt.yticks(np.arange(-1.2, 1.21, 0.4), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4)
    plt.axis([0, 50000, -1.2, 1.2])

    fig.savefig(out_path + "c_margin_dist.pdf", bbox_inches='tight')


def plot_nuclear():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['nuclear_metric'], 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['nuclear_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['nuclear_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('NF_metric', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 8.1, 2), fontsize=30) # plot for figure 6 cifar10

    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=1)

    plt.axis([0, 200, 0, 8])

    fig.savefig(out_path + "resnet18-nuclear.pdf", bbox_inches='tight')


def plot_avg_cos_margin():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_sgd['avg_cos_margin'], 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_adam['avg_cos_margin'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['avg_cos_margin'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\bar{CM}$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-0.4, 1.21, 0.4), fontsize=30)

    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4)

    plt.axis([0, 200, -0.4, 1.2])

    fig.savefig(out_path + "resnet18-avg-cmargin.pdf", bbox_inches='tight')


def plot_part_cos_margin_distribution(k=100):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    ITICKS = np.arange(0, 6*(k//5), k//5)

    plt.plot(info_sgd['all_cos_margin'][-1][:k], 'r', marker='v', ms=16, markevery=k//5, linewidth=5, alpha=0.7)
    plt.plot(info_adam['all_cos_margin'][-1][:k], 'b', marker='o', ms=16, markevery=k//5, linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['all_cos_margin'][-1][:k], 'g', marker='s', ms=16, markevery=k//5, linewidth=5, alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel(r'$\mathcal{P}_{CM}$', fontsize=40)
    plt.xticks(ITICKS, fontsize=30)

    plt.yticks(np.arange(-1.2, 1.21, 0.4), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4)
    plt.axis([0, k, -1.2, 1.2])

    fig.savefig(out_path + "c_margin_dist_part.pdf", bbox_inches='tight')


def main():
    plot_collapse()
    plot_ETF()
    plot_WH_relation()
    plot_residual()

    plot_train_acc()
    plot_test_acc()

    plot_cos_margin_distribution()

    plot_nuclear()
    plot_avg_cos_margin()

    plot_part_cos_margin_distribution(k=1000)


if __name__ == "__main__":
    main()