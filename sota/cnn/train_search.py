import json
import os
import sys

sys.path.insert(0, os.path.expanduser('~/workspace/darts-SAM'))

from imagenet16 import ImageNet16
import time
import glob
import numpy as np
import torch
import optimizers.darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from sota.cnn.model_search import Network, beta_decay_scheduler
#from optimizers.dartsminus.model_search import Network as NetworkDartsMinus
from optimizers.darts.architect import Architect
from sota.cnn.spaces import spaces_dict

from attacker.perturb import Linf_PGD_alpha, Random_alpha

from copy import deepcopy
from numpy import linalg as LA

# from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

import wandb

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='/remote-home/source/share/dataset', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')

# Our arguments
parser.add_argument('--wandb', type=str2bool, default=False, help='use one-step unrolled validation loss')
parser.add_argument('--nasbench', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--betadecay', type=str2bool, default=False, help='use beta-darts regularization')
parser.add_argument('--sam', type=str2bool, default=False, help='use sam update rule')
parser.add_argument('--unrolled', type=str2bool, default=False, help='use one-step unrolled validation loss')
parser.add_argument('--rho_alpha_sam', type=float, default=1e-2, help='rho alpha for SAM update')
parser.add_argument('--epsilon_sam', type=float, default=1e-2, help='epsilon for SAM update')
parser.add_argument('--data_aug', type=str2bool, default=True, help='use data augmentation on validation set')
parser.add_argument('--w_nor', type=float, default=0.5, help='epsilon for beta regularization normal component')
parser.add_argument('--k_sam', type=int, default=1, help='Number of ascent steps for SAM')
parser.add_argument('--sgd_alpha', action='store_true', default=False, help='use sgd optim for arch encoding')
parser.add_argument('--auxiliary_skip', action='store_true', default=False, help='use aux operation in mixedop (darts-)')

args = parser.parse_args()

'''
args.save = '../../experiments/sota/{}/search-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.search_space, args.seed)

if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
if not args.perturb_alpha == 'none':
    args.save += '-alpha-' + args.perturb_alpha + '-' + str(args.epsilon_alpha)
args.save += '-' + str(np.random.randint(10000))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
'''

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
#writer = SummaryWriter(args.save + '/runs')

if args.wandb:
    wandb.init(
        # username or team name
        entity='flatnas',
        # set the wandb project where this run will be logged
        project=f"FlatDARTS-{args.dataset}-nasbench{args.nasbench}-data_aug",
        name=f"SAM_{args.sam}-BETADECAY_{args.betadecay}-WNOR_{args.w_nor}-UNROLLED_{args.unrolled}-DATA_AUG_{args.data_aug}-RHO_ALPHA_{args.rho_alpha_sam}-K_SAM_{args.k_sam}-SGD_ALPHA_{args.sgd_alpha}-SEED{args.seed}",
        # track hyperparameters and run metadata
        config={**vars(args)},
    )


if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif args.perturb_alpha == 'random':
        perturb_alpha = Random_alpha

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], auxiliary_skip=args.auxiliary_skip)

    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    elif args.dataset == 'imagenet16':
        train_transform, valid_transform = utils._data_transforms_imagenet16(args)
        train_data = ImageNet16(root=args.data, train=True, transform=train_transform, use_num_of_class_only=n_classes)
        valid_data = ImageNet16(root=args.data, train=True, transform=valid_transform, use_num_of_class_only=n_classes)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    if 'debug' in args.save:
        split = args.batch_size
        num_train = 2 * args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader( 
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    best_loss= float('inf')
    best_epoch=0
    best_acc=0
    best_genotype=None

    patience=10

    for epoch in range(args.epochs):

        lr = scheduler.get_last_lr()[0]
        if args.cutout:
            # increase the cutout probability linearly throughout search
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        
        if args.perturb_alpha:
            epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
            logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        #print(F.softmax(model.alphas_normal, dim=-1))
        #print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, 
                                         perturb_alpha, epsilon_alpha)
        
        if args.sgd_alpha:
            sam_alpha = architect.get_arch_lr()

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)

        if args.betadecay:
            beta_loss = architect._beta_loss()        

        if args.wandb:
            wandb.log({"metrics/train_acc": train_acc, 
                        "metrics/val_acc": valid_acc,
                        "metrics/train_loss": train_obj,
                        "metrics/val_loss": valid_obj})
            if args.betadecay:
                wandb.log({"metrics/beta_loss": beta_loss})
            if args.sgd_alpha:
                wandb.log({"metrics/arch_lr": sam_alpha})
            
        logging.info("Train acc: %.2f, Val acc: %.2f", train_acc, valid_acc)
        logging.info("Train loss: %.2f, Val loss: %.2f", train_obj, valid_obj)
        if args.betadecay:
            logging.info("Beta loss: %.2f", beta_loss)
        if args.sgd_alpha:
            logging.info("Arch lr: %.2f", sam_alpha)

        if valid_obj < best_loss:
            logging.info('Best model found at epoch %d', epoch)
            best_loss = valid_obj
            best_epoch = epoch
            best_acc = valid_acc
            best_genotype = genotype
            utils.save(model, os.path.join(args.save, 'weights.pt'))

        # early stopping
        if (epoch - best_epoch) > patience:
            logging.info('Early stopping at epoch %d', epoch)
            break

        scheduler.step()

        if args.auxiliary_skip:
            beta_decay_scheduler.step(epoch)
            logging.info('Beta: %f', beta_decay_scheduler.decay_rate)

    #writer.close()
    # Info about best searched model

    logging.info('Best model found at epoch %s', best_epoch) 
    logging.info('Best genotype: %s', best_genotype)
    logging.info('Best validation loss: %f', best_loss)
    logging.info('Best validation accuracy: %f', best_acc)

    #genotype_dict = genotype_to_dict(best_genotype)

    # Save to a file
    #with open(os.path.join(args.save,'genotype.json'), 'w') as f:
    #    json.dump(genotype_dict, f, indent=4)

    stats = {'best_genotype': str(best_genotype), 'val-acc': best_acc, 'val-loss': best_loss, 'best_epoch': best_epoch}
    with open(os.path.join(args.save, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

# Convert Genotype to a serializable dictionary
def genotype_to_dict(genotype):
    # Create a dictionary with mandatory fields
    genotype_dict = {
        'normal': genotype.normal,
        'normal_concat': list(genotype.normal_concat)
    }
    
    # Check if 'reduce' and 'reduce_concat' attributes exist
    if hasattr(genotype, 'reduce'):
        genotype_dict['reduce'] = genotype.reduce
    if hasattr(genotype, 'reduce_concat'):
        genotype_dict['reduce_concat'] = list(genotype.reduce_concat)
    
    return genotype_dict


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, perturb_alpha, epsilon_alpha):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, step, unrolled=args.unrolled)
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        #print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()

        # perturb on alpha
        #print('after softmax', model.arch_parameters())
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        #print('after perturb', model.arch_parameters())

        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.restore_arch_parameters()
        #print('after restore', model.arch_parameters())

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break

    return  top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if 'debug' in args.save:
                    break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 
