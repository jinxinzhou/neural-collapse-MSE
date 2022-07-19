import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/')

PATH_TO_INFO_M_1_k_1 = os.path.join(PATH_TO_INFO, 'mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_1_k_1/'+'info.pkl')
PATH_TO_INFO_M_3_k_1 = os.path.join(PATH_TO_INFO, 'mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_3_k_1/'+'info.pkl')
PATH_TO_INFO_M_5_k_1 = os.path.join(PATH_TO_INFO, 'mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_5_k_1/'+'info.pkl')
PATH_TO_INFO_M_7_k_1 = os.path.join(PATH_TO_INFO, 'mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_7_k_1/'+'info.pkl')
PATH_TO_INFO_M_10_k_1 = os.path.join(PATH_TO_INFO, 'mini_imagenet_RMSE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_M_10_k_1/'+'info.pkl')
#
out_path = os.path.join(os.path.dirname(PATH_TO_INFO), '/imgs/rmse/M_imgs/')
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)
#
with open(PATH_TO_INFO_M_1_k_1, 'rb') as f:
    info_M_1_k_1 = pickle.load(f)

with open(PATH_TO_INFO_M_3_k_1, 'rb') as f:
    info_M_3_k_1 = pickle.load(f)

with open(PATH_TO_INFO_M_5_k_1, 'rb') as f:
    info_M_5_k_1 = pickle.load(f)
#
with open(PATH_TO_INFO_M_7_k_1, 'rb') as f:
    info_M_7_k_1 = pickle.load(f)

with open(PATH_TO_INFO_M_10_k_1, 'rb') as f:
    info_M_10_k_1 = pickle.load(f)


XTICKS = [0, 50, 100, 150, 200]


def plot_collapse():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['collapse_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['collapse_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['collapse_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['collapse_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['collapse_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 12.1, 4), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)
    plt.axis([0, 200, -0.4, 12])

    fig.savefig(out_path  + "resnet18-NC1.pdf", bbox_inches='tight')


def plot_ETF():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['ETF_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['ETF_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['ETF_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['ETF_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['ETF_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(-0.2, 1.21, .2), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)

    plt.axis([0, 200, -0.02, 1.2])

    fig.savefig(out_path + "resnet18-NC2.pdf", bbox_inches='tight')


def plot_WH_relation():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['WH_relation_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['WH_relation_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['WH_relation_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['WH_relation_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['WH_relation_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 1.21, 0.2), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)

    plt.axis([0, 200, 0, 1.2])

    fig.savefig(out_path + "resnet18-NC3.pdf", bbox_inches='tight')


def plot_residual():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['Wh_b_relation_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['Wh_b_relation_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['Wh_b_relation_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['Wh_b_relation_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['Wh_b_relation_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)

    plt.axis([0, 200, 0, 1.0])

    fig.savefig(out_path + "resnet18-NC4.pdf", bbox_inches='tight')


def plot_train_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['train_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['train_acc1'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['train_acc1'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['train_acc1'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['train_acc1'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(40, 110, 20), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)

    plt.axis([0, 200, 20, 102])

    fig.savefig(out_path + "resnet18-train-acc.pdf", bbox_inches='tight')


def plot_test_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['test_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['test_acc1'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['test_acc1'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['test_acc1'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['test_acc1'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(20, 100.1, 10), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)

    plt.axis([0, 200, 20, 100])

    fig.savefig(out_path +"resnet18-test-acc.pdf", bbox_inches='tight')


def plot_cos_margin_distribution():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    ITICKS = [0, 10000, 20000, 30000, 40000, 50000]
    # ------------------------------------- plot for figure 6 ----------------------------------------------------------
    plt.plot(info_M_1_k_1['all_cos_margin'][-1], 'c', marker='v', ms=16, markevery=10000, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['all_cos_margin'][-1], 'b', marker='o', ms=16, markevery=10000, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['all_cos_margin'][-1], 'g', marker='s', ms=16, markevery=10000, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['all_cos_margin'][-1], 'r', marker='X', ms=16, markevery=10000, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['all_cos_margin'][-1], 'purple', marker='>', ms=16, markevery=10000, linewidth=5,
             alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel('Cos Margin Distribution', fontsize=40)
    plt.xticks(ITICKS, fontsize=30)

    plt.yticks(np.arange(-1.2, 1.3, 0.4), fontsize=30)
    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)
    plt.axis([0, 50000, -1.2, 1.3])

    fig.savefig(out_path + "c_margin_dist.pdf", bbox_inches='tight')


def plot_nuclear():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['nuclear_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['nuclear_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['nuclear_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['nuclear_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['nuclear_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Avg. class NF_metric', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 6.1, 1), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)

    plt.axis([0, 200, 0, 6])

    fig.savefig(out_path + "resnet18-nuclear.pdf", bbox_inches='tight')


def plot_avg_cos_margin():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_M_1_k_1['avg_cos_margin'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['avg_cos_margin'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['avg_cos_margin'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['avg_cos_margin'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['avg_cos_margin'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Avg. Cos Margin', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-0.4, 1.21, 0.4), fontsize=30)

    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)

    plt.axis([0, 200, -0.4, 1.2])

    fig.savefig(out_path + "resnet18-avg-cmargin.pdf", bbox_inches='tight')


def plot_part_cos_margin_distribution(k=100):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    ITICKS = np.arange(0, 6*(k//5), k//5)

    plt.plot(info_M_1_k_1['all_cos_margin'][-1][:k], 'c', marker='v', ms=16, markevery=k//5, linewidth=5, alpha=0.7)
    plt.plot(info_M_3_k_1['all_cos_margin'][-1][:k], 'b', marker='o', ms=16, markevery=k//5, linewidth=5, alpha=0.7)
    plt.plot(info_M_5_k_1['all_cos_margin'][-1][:k], 'g', marker='s', ms=16, markevery=k//5, linewidth=5, alpha=0.7)
    plt.plot(info_M_7_k_1['all_cos_margin'][-1][:k], 'r', marker='v', ms=16, markevery=k//5, linewidth=5, alpha=0.7)
    plt.plot(info_M_10_k_1['all_cos_margin'][-1][:k], 'purple', marker='>', ms=16, markevery=k // 5,
             linewidth=5, alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel(r'$\mathcal{P}_{CM}$', fontsize=40)
    plt.xticks(ITICKS, fontsize=30)

    plt.yticks(np.arange(-1.2, 1.21, 0.4), fontsize=30)
    plt.legend([r'$\alpha=1, M=1$', r'$\alpha=1, M=3$', r'$\alpha=1, M=5$', r'$\alpha=1, M=7$', r'$\alpha=1, M=10$'], fontsize=30)
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