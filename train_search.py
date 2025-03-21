from collections import OrderedDict
import json
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.expanduser('~'),'workspace/darts-SAM')) 

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
from nasbench201.model import Network as BenchNetwork, beta_decay_scheduler
from optimizers.darts.model_search2 import Network as DARTSNetwork
from optimizers.dartsminus.model_search import Network as DARTSMINUSNetwork
from optimizers.darts.architect import Architect

from attacker.perturb import Linf_PGD_alpha, Random_alpha

from copy import deepcopy
from numpy import linalg as LA

#from torch.utils.tensorboard import SummaryWriter
#from nas_201_api import NASBench201API as API
from nasbench201.archive import NASBench201
from nasbench201.genotypes import BENCH_PRIMITIVES, Structure
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
parser.add_argument('--data', type=str, default='../datasets/cifar10', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')  # DARTS: 3e-4  RDARTS: 81e-4
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--device', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--n_cells', type=int, default=8, help='total number of cells')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', type=str2bool, default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
parser.add_argument('--wandb', type=str2bool, default=False, help='use one-step unrolled validation loss')
parser.add_argument('--nasbench', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--betadecay', type=str2bool, default=False, help='use beta-darts regularization')
parser.add_argument('--sam', type=str2bool, default=False, help='use sam update rule')
parser.add_argument('--rho_alpha_sam', type=float, default=1e-2, help='rho alpha for SAM update')
parser.add_argument('--epsilon_sam', type=float, default=1e-2, help='epsilon for SAM update')
parser.add_argument('--flood_level', type=float, default=0.0, help='flood level for weight regularization')
parser.add_argument('--data_aug', type=str2bool, default=True, help='use data augmentation on validation set')
parser.add_argument('--sgd_alpha', type=str2bool, default=False, help='lookbehind optimizer for alpha')
parser.add_argument('--k_sam', type=int, default=1, help='lookbehind steps for alpha')
parser.add_argument('--method', type=str, default='darts', help='method to use')
parser.add_argument('--auxiliary_skip', action='store_true', default=False, help='use aux operation in mixedop (darts-)')
parser.add_argument('--forward_mode', type=str, default='default', help='forward mode for enabling pcdarts')

args = parser.parse_args()

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
        name=f"SAM_{args.sam}-BETADECAY_{args.betadecay}-UNROLLED_{args.unrolled}-DATA_AUG_{args.data_aug}-RHO_ALPHA_{args.rho_alpha_sam}-SEED_{args.seed}",
        # track hyperparameters and run metadata
        config={**vars(args)},
    )


if args.dataset == 'cifar10':
    n_classes = 10
    dataset = 'cifar10'
elif args.dataset == 'cifar100':
    n_classes = 100
    dataset = 'cifar100'
elif args.dataset == 'imagenet16':
    n_classes = 120
    dataset = 'ImageNet16-120'

def flatten_tuples(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.device)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.device)
    logging.info("args = %s", args)

    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif args.perturb_alpha == 'random':
        perturb_alpha = Random_alpha
    
    #api = API('/remote-home/share/share/dataset/NAS-Bench-201-v1_0-e61699.pth')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #model = Network(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion)  # N=5/1/3

    if not args.nasbench:
        if args.method == 'darts':
            model = DARTSNetwork(args.init_channels, n_classes, args.n_cells, criterion)
        else:
            print('Using DARTSMINUS')
            model = DARTSMINUSNetwork(args.init_channels, n_classes, args.n_cells, criterion)
        
    else:
        #stages = 3
        #cells = 5
        model = BenchNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, auxiliary_skip=args.auxiliary_skip,
                             forward_mode=args.forward_mode)

    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    input_shape=(3, 32, 32)
    #avg_macs = model.compute_network_cost(input_shape)
    #logging.info("MACs: %f", avg_macs)

    optimizer = torch.optim.SGD(
        model.get_weights(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    
    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='train', download=True, transform=valid_transform)
    elif args.dataset == 'imagenet16':
        train_transform, valid_transform = utils._data_transforms_imagenet16(args)
        train_data = ImageNet16(root=args.data, train=True, transform=train_transform, use_num_of_class_only=n_classes)
        valid_data = ImageNet16(root=args.data, train=True, transform=valid_transform, use_num_of_class_only=n_classes)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))
    split_train = int(np.floor(0.5 * num_train))
    split_valid = int(np.floor(0.5 * num_train))
    print('num_train =', num_train, 'split_train =', split_train, 'split_valid =', split_valid)
    if 'debug' in args.save:
        split = args.batch_size
        num_train = 2 * args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split_train]),
        pin_memory=True)

    if args.data_aug:
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split_valid:num_train]),
            pin_memory=True)
    else:
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split_valid:num_train]),
            pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    bench=NASBench201(dataset=dataset)

    best_loss= float('inf')
    best_epoch=0
    best_acc=0
    best_genotype=None

    patience=10
    
    for epoch in range(args.epochs):
        #scheduler.step()
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
        if args.nasbench:
            genotype=genotype.to_genotype()
        logging.info('genotype = %s', genotype)

        print(model.show_alphas())

        
        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, 
                                         perturb_alpha, epsilon_alpha, epoch)
        
        scheduler.step() # scheduler step must be done after optimizer.step() in latest pytorch versions
        if args.auxiliary_skip:
            beta_decay_scheduler.step(epoch)
            logging.info('Beta: %f', beta_decay_scheduler.decay_rate)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        
        if args.betadecay and not args.nasbench:
            beta_loss = architect._beta_loss()        

        if args.wandb and args.beta_decay:
            if not args.nasbench:
                wandb.log({"metrics/train_acc": train_acc, 
                        "metrics/val_acc": valid_acc,
                        "metrics/train_loss": train_obj,
                        "metrics/val_loss": valid_obj})
            else:
                wandb.log({"metrics/train_acc": train_acc, 
                        "metrics/val_acc": valid_acc,
                        "metrics/train_loss": train_obj,
                        "metrics/val_loss": valid_obj,
                        "metrics/beta_loss": beta_loss})
                logging.info("Beta loss: %.2f", beta_loss)
            
        logging.info("Train acc: %.2f, Val acc: %.2f", train_acc, valid_acc)
        logging.info("Train loss: %.2f, Val loss: %.2f", train_obj, valid_obj)
    
        if valid_obj < best_loss:
            logging.info('Best model found at epoch %d', epoch)
            best_loss = valid_obj
            best_epoch = epoch
            best_acc = valid_acc
            best_genotype = genotype

        if args.nasbench:
            cell_encode = translate_genotype_to_encode(genotype)
            decode = bench.decode(cell_encode)
            info = bench.get_info_from_arch(decode)
            #results = {'val-acc': info['val-acc'], 'test-acc': info['test-acc'], 'flops': info['flops'], 'params': info['params']}

            if args.wandb:
                wandb.log({"metrics/val_acc_nasbench": info['val-acc'], "metrics/test_acc_nasbench": info['test-acc']})
            logging.info('NASBench201 val acc: %.2f, test acc: %.2f', info['val-acc'], info['test-acc'])
            
        if not args.unrolled:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'alpha': model.arch_parameters()
            }, False, args.save)

        # early stopping
        if (epoch - best_epoch) > patience:
            logging.info('Early stopping at epoch %d', epoch)
            break

    # Info about best searched model

    logging.info('Best model found at epoch %s', best_epoch) 
    logging.info('Best genotype: %s', best_genotype)
    logging.info('Best validation loss: %f', best_loss)
    logging.info('Best validation accuracy: %f', best_acc)
    
    if args.nasbench:
            cell_encode = translate_genotype_to_encode(best_genotype)
            decode = bench.decode(cell_encode)
            datasets=['cifar10', 'cifar100', 'ImageNet16-120']
            stats = {'best_genotype': str(best_genotype)}
            for ds in datasets:
                bench.dataset = ds
                info = bench.get_info_from_arch(decode)
                logging.info('BEST NASBench201 val acc: %.2f, test acc: %.2f', info['val-acc'], info['test-acc'])
                stats[f'{ds}_val']=info['val-acc']
                stats[f'{ds}_test']=info['test-acc']
            
            with open(os.path.join(args.save,'stats.json'), 'w') as f:
                json.dump(stats, f, indent=4)

    genotype_dict = genotype_to_dict(best_genotype)

    # Save to a file
    with open(os.path.join(args.save,'genotype.json'), 'w') as f:
        json.dump(genotype_dict, f, indent=4)

    #call_training_script(n_classes,args)

def call_training_script(n_classes,args):
    # Prepare the command to call train.py and save command script in the same directory

    bash_file = ['#!/bin/bash']
    cfg =OrderedDict()
    
    cfg['dataset'] = args.dataset
    cfg['data'] = args.data
    cfg['device'] = args.device
    cfg['output_path'] = args.save
    cfg['n_classes'] = n_classes
    cfg['epochs'] = 600
    cfg['batch_size'] = 96
    cfg['learning_rate'] = 0.1
    cfg['weight_decay'] = 0.0005
    cfg['momentum'] = 0.9
    cfg['drop_path_prob'] = 0.2
    cfg['auxiliary'] = True
    cfg['auxiliary_weight'] = 0.4
    cfg['cutout'] = True
    cfg['res'] = 32
    cfg['optim'] = 'SGD'
    cfg['eval_test'] = True
    cfg['nesterov'] = True
    cfg['seed'] = args.seed

    execution_line = "python train.py"
    for k, v in cfg.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    
    bash_file.append(execution_line)
    
    with open(os.path.join(args.save, 'run_bash.sh'), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)

    subprocess.call("sh {}/run_bash.sh".format(args.save), shell=True)


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


def write_array_to_file(arr, file_path):
    with open(file_path, 'w') as file:
        array_str = ','.join(map(str, arr))  # Join integers into a string separated by comma
        file.write(array_str)

def translate_genotype_to_encode(genotype):
    dag_integers = []
    for node_op, _ in genotype.normal:
        dag_integers.append(BENCH_PRIMITIVES.index(node_op))
    return dag_integers


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, perturb_alpha, epsilon_alpha, epoch):
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

        architect.step(input, target, input_search, target_search, lr, optimizer, epoch, unrolled=args.unrolled)
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        # print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()

        # perturb on alpha
        # print('after softmax', model.arch_parameters())
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        # print('after perturb', model.arch_parameters())

        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.restore_arch_parameters()
        # print('after restore', model.arch_parameters())

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
