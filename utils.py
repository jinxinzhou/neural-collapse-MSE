import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from optim_local.sdlbfgs import SdLBFGS

def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9,
                  'lr': args.lr,
                  'weight_decay': args.weight_decay
        }
    elif args.optimizer == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
    elif args.optimizer == 'LBFGS':
        optimizer_function = optim.LBFGS
        kwargs = {'lr': args.lr,
                  'history_size': args.history_size,
                  'line_search_fn': 'strong_wolfe'
        }
    elif args.optimizer == 'SdLBFGS':
        optimizer_function = SdLBFGS
        kwargs = {'lr': args.lr,
                  'history_size': args.history_size,
                  'lr_decay': False,
                  }
    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.patience,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'cos_anneal_warmup':
        schduler = lrs.CosineAnnealingWarmRestarts(
            my_optimizer,
            T_0=args.epochs,
            eta_min=0.0001
        )

    return scheduler


def make_criterion(args):
    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss == 'RescaledMSE':
        criterion = Rescale_Square_Loss(num_classes=args.num_classes, M=args.M, k=args.k)

    return criterion


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def print_and_save(text_str, file_stream):
    print(text_str)
    print(text_str, file=file_stream)


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Rescale_Square_Loss(nn.Module):
    def __init__(self, num_classes=10, k=1, M=1,  reduction = 'mean'):
        """
        param k: k rescales the loss value at the true label,
        param M: M rescales the one-hot encoding
        """
        super(Rescale_Square_Loss, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.M = M
        self.reduction = reduction

    def forward(self, output, target):
        # First, transfer class labels to one-hot targets
        one_hot_target = nn.functional.one_hot(target.type(torch.LongTensor), num_classes=self.num_classes).type(torch.FloatTensor).to(target.device)
        diff = (output - self.M * one_hot_target) ** 2
        diff = diff * (1-one_hot_target) + diff * one_hot_target * self.k
        loss_tmp = torch.mean(diff, dim=1, keepdim=True)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss
