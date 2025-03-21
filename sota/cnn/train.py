import json
import os
import sys

home_dir = os.path.expanduser('~')
sys.path.insert(0, os.path.join(home_dir, 'workspace', 'darts-SAM'))
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
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import sota.cnn.genotypes as genotypes

from torch.autograd import Variable
from sota.cnn.model_cifar import NetworkCIFAR as Network
# from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import wandb


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/remote-home/source/share/dataset',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default=None, help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--wandb', action='store_true', default=False, help='use wandb')
parser.add_argument('--train_limit', type=float, default=0.0, help='training loss limit')

# Add a resume argument to argparse
parser.add_argument('--resume', type=str, default=None, help='path to resume checkpoint')

args = parser.parse_args()

# Define the save_checkpoint function
def save_checkpoint(state, save_path, epoch):
    filename = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pt')
    torch.save(state, filename)
    logging.info(f"Checkpoint saved at epoch {epoch}: {filename}")

# Define the load_checkpoint function
def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if not os.path.isfile(checkpoint_path):
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_valid_obj = checkpoint['best_valid_obj']
    logging.info(f"Checkpoint loaded: {checkpoint_path} (epoch {start_epoch})")
    return start_epoch, best_valid_obj

'''
args.save = '../../experiments/sota/{}/eval-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
if args.auxiliary:
    args.save += '-auxiliary-' + str(args.auxiliary_weight)
args.save += '-' + str(np.random.randint(10000))
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
        project=f"FlatDARTS-TRAIN-{args.dataset}",
        name=f"TRAIN_ARCH_{args.arch}",
        # track hyperparameters and run metadata
        config={**vars(args)},
    )

if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'cifar10':
    n_classes = 10
elif args.dataset == 'ImageNet16': #imagenet16-120
    n_classes = 120

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

    if args.arch is None:
        #read arch from genotype json file
        with open(os.path.join(args.save, 'genotype.json'), 'r') as f:
            genotype = json.load(f)
        print(genotype)
        #transform dict to genotype
        genotype = genotypes.Genotype(normal=genotype['normal'], normal_concat=genotype['normal_concat'],
                                      reduce=genotype['reduce'], reduce_concat=genotype['reduce_concat'])
    else:
        genotype = eval("genotypes.%s" % args.arch)
        
    logging.info(genotype)
    #if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    model = Network(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
    #else: 
    #    model = NetworkImageNet(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
    elif args.dataset == 'ImageNet16':
        train_transform, valid_transform = utils._data_transforms_imagenet16(args)
        train_data = ImageNet16(root=args.data, train=True, transform=train_transform, use_num_of_class_only=n_classes)
        valid_data = ImageNet16(root=args.data, train=False, transform=valid_transform, use_num_of_class_only=n_classes)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))
    
    best_valid_obj = float('inf')  # Initialize the best validation loss as infinity
    best_valid_acc = 0.0  # Initialize the best validation accuracy as 0.0
    best_train_acc = 0.0  # Initialize the best training accuracy as 0.0
    best_train_obj = float('inf')  # Initialize the best training loss as infinity
    patience = 100  # Patience for early stopping
    counter=0

    # Load checkpoint if specified
    start_epoch = 0
    best_valid_obj = float('inf')
    if args.resume:
        start_epoch, best_valid_obj = load_checkpoint(args.resume, model, optimizer, scheduler)

    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_last_lr()[0]
        if args.cutout:
            # increase the cutout probability linearly throughout search
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * \
                epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)

        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        if args.wandb:
            wandb.log({"metrics/train_acc": train_acc, 
                       "metrics/val_acc": valid_acc,
                       "metrics/train_loss": train_obj,
                       "metrics/val_loss": valid_obj})

        # Save the best model weights
        if valid_obj < best_valid_obj:
            best_valid_obj = valid_obj
            best_valid_acc = valid_acc
            best_train_acc = train_acc
            best_train_obj = train_obj
            utils.save(model, os.path.join(args.save, 'best_weights.pt'))
            counter=0
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
        
        # Check if the training loss is below the threshold to stop training
        if train_obj < args.train_limit:
            logging.info('Training loss has fallen below the threshold. Stopping training.')
            break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_obj': best_valid_obj
            }, args.save, epoch + 1)

        scheduler.step()


    #writer.close()

    #Save tuple (config, stats) to a json file
    
    #Save both best and current values
    best_train_acc = np.round(best_train_acc.item(), 3)
    best_train_obj = np.round(best_train_obj.item(), 3)
    best_valid_acc = np.round(best_valid_acc.item(), 3)
    best_valid_obj = np.round(best_valid_obj.item(), 3)

    train_acc = np.round(train_acc.item(), 3)
    train_obj = np.round(train_obj.item(), 3)
    valid_acc = np.round(valid_acc.item(), 3)
    valid_obj = np.round(valid_obj.item(), 3)

    genotype = {'normal': genotype.normal, 'normal_concat': list(genotype.normal_concat),
                'reduce': genotype.reduce, 'reduce_concat': list(genotype.reduce_concat)}

    with open(os.path.join(args.save, 'stats.json'), 'w') as f:
        json.dump({'config': genotype, 'train_acc': train_acc, 'train_loss': train_obj, 'val_acc': valid_acc, 'val_loss': valid_obj,
        'best_train_acc': best_train_acc, 'best_train_obj': best_train_obj, 'best_valid_acc': best_valid_acc, 'best_valid_obj': best_valid_obj}, f)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
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
