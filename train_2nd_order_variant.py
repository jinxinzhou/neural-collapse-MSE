import sys

import torch

import models
from utils import *
from args import parse_train_args
from datasets import make_dataset


def weight_decay(args, model):

    penalty = 0
    for p in model.parameters():
        if p.requires_grad:
            penalty += args.penalty * args.weight_decay * torch.norm(p) ** 2

    return penalty.to(args.device)

def get_grad(args, inputs, targets, model, criterion, optimizer):
    """
    Computes objective and gradient of neural network over data sample.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere https://github.com/hjmshi/PyTorch-LBFGS/blob/4cd150ff0e4c4dd2183c421170eb0d3b56eb5885/functions/utils.py#L81

    Inputs:
        args (object): user-defined arguments for training
        X_Sk (tensor): set of training examples over sample Sk
        y_Sk (tensor): set of training labels over sample Sk
        optimizer (Optimizer): the PBQN optimizer
        #opfun (callable): computes forward pass over network over sample Sk
        #ghost_batch (int): maximum size of effective batch (default: 128)
    Outputs:
        grad (tensor): stochastic gradient over sample Sk
        obj (tensor): stochastic function value over sample Sk
    """
    Sk_size = inputs.shape[0]
    ghost_batch = args.gost_batch

    batch_loss = torch.tensor(0, dtype=torch.float).to(args.device)

    optimizer.zero_grad()

    # loop through revelant data
    for idx in np.array_split(np.arange(Sk_size), max(int(Sk_size/ghost_batch), 1)):

        X_Sk = inputs[idx]
        y_Sk = targets[idx]

        outputs_Sk = model(X_Sk)
        # define sample loss and perform forward-backward pass
        if args.loss == 'CrossEntropy':
            loss_Sk = criterion(outputs_Sk[0], y_Sk) + weight_decay(args, model)
        elif args.loss == 'MSE':
            loss_Sk = criterion(outputs_Sk[0], nn.functional.one_hot(y_Sk).type(torch.FloatTensor).to(args.device)) \
                   + weight_decay(args, model)

        loss_Sk /= len(idx)/Sk_size
        loss_Sk.backward()

        # accumulate the total batch loss
        batch_loss += loss_Sk

    # gather flat gradient
    grad = optimizer._gather_flat_grad()
    return grad, batch_loss

def trainer(args, model, trainloader, epoch_id, criterion, optimizer, logfile):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTraining Epoch: [%d | %d]' % (epoch_id + 1, args.epochs), logfile)
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.train()

        # compute initial gradient and objective
        grad, current_loss = get_grad(args, inputs, targets, model, criterion, optimizer)

        # two-loop recursion to compute search direction
        p = optimizer.two_loop_recursion(-grad)

        def closure():

            optimizer.zero_grad()

            Sk_size = inputs.shape[0]
            ghost_batch = args.gost_batch

            loss_fn = torch.tensor(0, dtype=torch.float).to(args.device)

            for idx in np.array_split(np.arange(Sk_size), max(int(Sk_size/ghost_batch), 1)):
                X_Sk = inputs[idx]
                y_SK = targets[idx]

                outputs_Sk = models[X_Sk]

                if args.loss == 'CrossEntropy':
                    loss_Sk = criterion(outputs_Sk[0], y_SK) + weight_decay(args, model)
                elif args.loss == 'MSE':
                    loss_Sk = criterion(outputs_Sk[0],
                                     nn.functional.one_hot(y_SK).type(torch.FloatTensor).to(args.device)) \
                           + weight_decay(args, model)

                loss_fn += loss_Sk * (len(idx)/Sk_size)

            return loss_fn
        # perform line search step
        kwargs = {'closure': closure, 'current_loss':current_loss}
        grad  = optimizer.step(p, grad, options=kwargs)[1]

        # curvature update
        optimizer.curvature_update(grad)

        # measure accuracy and record loss
        model.eval()
        outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))

        if args.loss == 'CrossEntropy':
            loss = criterion(outputs[0], targets) + weight_decay(args, model)
        elif args.loss == 'MSE':
            loss = criterion(outputs[0], nn.functional.one_hot(targets).type(torch.FloatTensor).to(args.device)) \
                   + weight_decay(args, model)

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
                           (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)


def train(args, model, trainloader):

    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)

    logfile = open('%s/log.txt' % (args.save_path), 'w')

    for epoch_id in range(args.epochs):

        trainer(args, model, trainloader, epoch_id, criterion, optimizer, logfile)
        torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")

    logfile.close()


def main():
    args = parse_train_args()

    if args.optimizer != 'LBFGS-variant':
        sys.exit('Support for training with LBFGS-variant only!')

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size)

    model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias).to(device)
    print('# of model parameters: ' + str(count_network_parameters(model)))

    train(args, model, trainloader)


if __name__ == "__main__":
    main()