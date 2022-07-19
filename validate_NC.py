import os.path
import sys
import pickle

import numpy as np
import torch
from PIL import Image
import scipy.linalg as scilin

import models
from utils import *
from args import parse_eval_args
from datasets import make_dataset


class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def split_array(input_array, batchsize=128):
    input_size = input_array.shape[0]
    num_splits, res_splits = input_size // batchsize, input_size % batchsize
    output_array_list = list()
    if res_splits == 0:
        output_array_list = np.split(input_array, batchsize, axis=0)
    else:
        for i in range(num_splits):
            output_array_list.append(input_array[i * batchsize:(i + 1) * batchsize])

        output_array_list.append(input_array[num_splits * batchsize:])

    return output_array_list


def compute_info(args, model, fc_features, dataloader, last_epoch=False):
    num_data = 0
    mu_G = 0
    mu_c_dict = dict()
    num_class_dict = dict()
    before_class_dict = dict()
    after_class_dict = dict()
    last_epoch_false_img_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            # outputs = model(inputs)
            outputs, fea = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
                before_class_dict[y] = [features[b, :].detach().cpu().numpy()]
                after_class_dict[y] = [outputs[b, :].detach().cpu().numpy()]
                num_class_dict[y] = 1
            else:
                mu_c_dict[y] += features[b, :]
                before_class_dict[y].append(features[b, :].detach().cpu().numpy())
                after_class_dict[y].append(outputs[b, :].detach().cpu().numpy())
                num_class_dict[y] = num_class_dict[y] + 1

        num_data += targets.shape[0]

        if last_epoch:
            false_pred = (torch.max(outputs, dim=1)[1] != targets)
            false_targets = targets[false_pred]
            false_imgs = inputs[false_pred, :, :, :]
            mean = torch.Tensor([0.4914, 0.4822, 0.4465]).to(inputs.device).reshape(1, 3, 1, 1)
            std = torch.Tensor([0.2023, 0.1994, 0.2010]).to(inputs.device).reshape(1, 3, 1, 1)
            false_imgs = ((false_imgs * std + mean) * 255).permute(0, 2, 3, 1)
            for b in range(len(false_targets)):
                y = false_targets[b].item()
                if y not in last_epoch_false_img_dict:
                    last_epoch_false_img_dict[y] = [false_imgs[b, :].detach().cpu().numpy().astype(np.uint8)]
                else:
                    last_epoch_false_img_dict[y].append(false_imgs[b, :].detach().cpu().numpy().astype(np.uint8))

        prec1, prec5 = compute_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    mu_G /= num_data
    for i in range(len(mu_c_dict.keys())):
        mu_c_dict[i] /= num_class_dict[i]

    return mu_G, mu_c_dict, before_class_dict, after_class_dict, top1.avg, top5.avg, last_epoch_false_img_dict


def compute_Sigma_W(args, before_class_dict, mu_c_dict, batchsize=1024):
    num_data = 0
    Sigma_W = 0

    for target in before_class_dict.keys():
        class_feature_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        for features in class_feature_list:
            features = torch.from_numpy(features).to(args.device)
            Sigma_W_batch = (features - mu_c_dict[target].unsqueeze(0)).unsqueeze(2) * (
                    features - mu_c_dict[target].unsqueeze(0)).unsqueeze(1)
            Sigma_W += torch.sum(Sigma_W_batch, dim=0)
            num_data += features.shape[0]

    Sigma_W /= num_data
    return Sigma_W.detach().cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()


def compute_ETF(W):
    K = W.shape[0]
    W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    M = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()
    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H


def compute_Wh_b_relation(W, mu_G, b):
    Wh = torch.mv(W, mu_G.cuda())
    res_b = torch.norm(Wh + b, p='fro')
    return res_b.detach().cpu().numpy().item()


def compute_nuclear_frobenius(all_features):
    nf_metric_list = []
    for i in all_features:
        class_feature = np.array(all_features[i])
        _, s, _ = np.linalg.svd(class_feature)  # s is all singular values
        nuclear_norm = np.sum(s)
        frobenius_norm = np.linalg.norm(class_feature, ord='fro')
        nf_metric_class = nuclear_norm / frobenius_norm
        nf_metric_list.append(nf_metric_class)
    nf_metric = np.mean(nf_metric_list)
    return nf_metric


def compute_margin(args, before_class_dict, after_class_dict, W, b, mu_G, batchsize=1024):
    num_data = 0
    avg_cos_margin = 0
    all_cos_margin = list()

    W = W - torch.mean(W, dim=0, keepdim=True)

    for target in after_class_dict.keys():
        class_features_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        class_outputs_list = split_array(np.array(after_class_dict[target]), batchsize=batchsize)
        for i in range(len(class_outputs_list)):
            features, outputs = torch.from_numpy(class_features_list[i]).to(args.device), torch.from_numpy(
                class_outputs_list[i]).to(args.device)

            cos_outputs = (outputs - b.unsqueeze(0)) / (
                        torch.norm(features - mu_G.unsqueeze(0), dim=1, keepdim=True) * torch.norm(W.T, dim=0,
                                                                                                   keepdim=True))
            false_cos_outputs = cos_outputs.clone()
            false_cos_outputs[:, target] = -np.inf
            false_cos_targets = torch.argmax(false_cos_outputs, dim=1)

            cos_margin = cos_outputs[:, target] - torch.gather(false_cos_outputs, 1,
                                                               false_cos_targets.unsqueeze(1)).reshape(-1)
            all_cos_margin.append(cos_margin.detach().cpu().numpy())
            avg_cos_margin += torch.sum(cos_margin)

            num_data += features.shape[0]

    avg_cos_margin /= num_data
    all_cos_margin = np.sort(np.concatenate(all_cos_margin, axis=0))
    return avg_cos_margin.item(), all_cos_margin


def main():
    args = parse_eval_args()
    set_seed(manualSeed=args.seed)

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size)
    args.num_classes = num_classes

    model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc,
                                        fixdim=args.fixdim, SOTA=args.SOTA).to(device)

    fc_features = FCFeatures()
    model.fc.register_forward_pre_hook(fc_features)

    info_dict = {
        'collapse_metric': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'nuclear_metric': [],
        'avg_cos_margin': [],
        'all_cos_margin': [],
        'W': [],
        'b': [],
        # 'H': [],
        'mu_G_train': [],
        'mu_G_test': [],
        # 'mu_c_dict_train': [],
        # 'mu_c_dict_test': [],
        # 'before_class_dict_train': [],
        # 'after_class_dict_train': [],
        # 'before_class_dict_test': [],
        # 'after_class_dict_test': [],
        'train_acc1': [],
        'train_acc5': [],
        'test_acc1': [],
        'test_acc5': []
    }

    logfile = open('%s/test_log.txt' % (args.load_path), 'w')
    for i in range(args.epochs):
        model.load_state_dict(torch.load(args.load_path + 'epoch_' + str(i + 1).zfill(3) + '.pth'))
        model.eval()
        last_epoch = (i + 1 == args.epochs)

        for n, p in model.named_parameters():
            if 'fc.weight' in n:
                W = p
            if 'fc.bias' in n:
                b = p
        if not args.bias:
            b = torch.zeros((W.shape[0],), device=device)

        mu_G_train, mu_c_dict_train, before_class_dict_train, after_class_dict_train, \
        train_acc1, train_acc5, train_last_epoch_false_img_dict = compute_info(args, model, fc_features, trainloader,
                                                                               last_epoch=last_epoch)

        mu_G_test, mu_c_dict_test, before_class_dict_test, after_class_dict_test, \
        test_acc1, test_acc5, test_last_epoch_false_img_dict = compute_info(args, model, fc_features, testloader)

        Sigma_W = compute_Sigma_W(args, before_class_dict_train, mu_c_dict_train, batchsize=args.batch_size)
        # Sigma_W_test_norm = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, testloader, isTrain=False)
        Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

        collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
        ETF_metric = compute_ETF(W)
        WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train)
        Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b)
        #
        nf_metric_epoch = compute_nuclear_frobenius(before_class_dict_train)
        avg_cos_margin, all_cos_margin = compute_margin(args, before_class_dict_train, after_class_dict_train, W, b, mu_G_train,
                           batchsize=args.batch_size)

        info_dict['collapse_metric'].append(collapse_metric)
        info_dict['ETF_metric'].append(ETF_metric)
        info_dict['WH_relation_metric'].append(WH_relation_metric)
        info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)
        info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
        info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())
        # info_dict['mu_c_dict_train'].append(mu_c_dict_train)
        # info_dict['mu_c_dict_test'].append(mu_c_dict_test)
        # info_dict['before_class_dict_train'].append(before_class_dict_train)
        # info_dict['after_class_dict_train'].append(after_class_dict_train)
        # info_dict['before_class_dict_test'].append(before_class_dict_test)
        # info_dict['after_class_dict_test'].append(after_class_dict_test)

        info_dict['W'].append((W.detach().cpu().numpy()))
        # if args.bias:
        info_dict['b'].append(b.detach().cpu().numpy())

        # info_dict['nuclear_metric'].append(nf_metric_epoch)
        info_dict['avg_cos_margin'].append(avg_cos_margin)
        info_dict['all_cos_margin'].append(all_cos_margin)

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)

        print_and_save('[epoch: %d] | train top1: %.4f | train top5: %.4f | test top1: %.4f | test top5: %.4f ' %
                       (i + 1, train_acc1, train_acc5, test_acc1, test_acc5), logfile)

    with open(args.load_path + 'info.pkl', 'wb') as f:
        pickle.dump(info_dict, f)

    img_path = args.load_path + 'false_img/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in train_last_epoch_false_img_dict.keys():
        class_i_img_path = img_path + '%s/' % i
        if not os.path.exists(class_i_img_path):
            os.makedirs(class_i_img_path)
        for j in range(len(train_last_epoch_false_img_dict[i])):
            false_class_i_img_j = train_last_epoch_false_img_dict[i][j]
            false_class_i_img_j = Image.fromarray(false_class_i_img_j)
            false_class_i_img_j.save(class_i_img_path + '%s.png' % (j + 1))


if __name__ == "__main__":
    main()
