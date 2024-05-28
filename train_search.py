import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from darts.model_search import Network
from nasbench201.model_search import BenchNetwork
from nasbench201.nasbench201 import NASBench201
from architect import Architect
from utils import get_data_loaders
from genotypes import BENCH_PRIMITIVES

def translate_genotype_to_encode(genotype):
    dag_integers = []
    for node_op, _ in genotype.normal:
        dag_integers.append(BENCH_PRIMITIVES.index(node_op))
    return dag_integers

def write_array_to_file(arr, file_path):
    with open(file_path, 'w') as file:
        array_str = ','.join(map(str, arr))  # Join integers into a string separated by comma
        file.write(array_str)

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../datasets', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--save', type=str, default='../../results/darts', help='location of the exp')
parser.add_argument('--nasbench', action='store_true', default=False, help='use nasbench search space')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=False, help='use nesterov momentum')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers') 
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--betadecay', action='store_true', default=False, help='use beta-darts regularization')
parser.add_argument('--sam', action='store_true', default=False, help='use sam update rule')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--rho_alpha_sam', type=float, default=1e-2, help='rho alpha for SAM update')
parser.add_argument('--epsilon_sam', type=float, default=1e-2, help='epsilon for SAM update')
args = parser.parse_args()

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#CIFAR_CLASSES = 10
if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  if not args.nasbench:
    model = Network(args.init_channels, n_classes, args.layers, criterion)
  else:
    stages = 3
    cells = 5
    model = BenchNetwork(args.init_channels, n_classes, stages, cells, criterion)

  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay,
      nesterov=args.nesterov)
  
  res=32
  
  train_queue, valid_queue, test_queue = get_data_loaders(dataset='cifar10', batch_size=args.batch_size, threads=args.workers, 
                                            val_split=args.train_portion, img_size=res, augmentation=True, eval_test=True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  
  # Check if CUDA (GPU support) is available
  if torch.cuda.is_available():
      device = torch.device("cuda")  # Use GPU
      print("CUDA is available! Using GPU.")
  else:
      device = torch.device("cpu")   # Use CPU
      print("CUDA is not available. Using CPU.")

  if args.sam:
    logging.info('Using new SAM update rule')
  else:
    if args.betadecay:
        logging.info('Using Beta-DARTS')
    else:
        logging.info('Using original DARTS')

  architect = Architect(model, args)

  '''
  best_loss= float('inf')
  best_epoch=0
  best_acc=0
  best_genotype=None
  '''

  for epoch in range(args.epochs):
    #lr = scheduler.get_lr()[0]
    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    if args.cutout:
        # increase the cutout probability linearly throughout search
        train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
        logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                      train_transform.transforms[-1].cutout_prob)
    else:
        logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(model.show_alphas())

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
    logging.info('train_acc %f, train_loss %f', train_acc, train_obj)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f, val_loss %f', valid_acc, valid_obj)
    scheduler.step()

    '''
    if valid_obj < best_loss:
      logging.info('Best model found at epoch %d', epoch)
      best_loss = valid_obj
      best_epoch = epoch
      best_acc = valid_acc
      best_genotype = genotype
    '''
      
    #info nasbench
    
    if args.nasbench:
        bench = NASBench201(dataset=args.dataset)
        cell_encode = translate_genotype_to_encode(genotype)
        decode = bench.decode(cell_encode)
        info = bench.get_info_from_arch(decode)
        results = {'val-acc': info['val-acc'], 'test-acc': info['test-acc'], 'flops': info['flops'], 'params': info['params']}
        logging.info('nasbench info: %s', results)
    

  utils.save(model, os.path.join(args.save, 'weights.pt'))
  
  # Info about best searched model

  logging.info('Best model found', epoch) 
  logging.info('Best genotype: %s', genotype)
  logging.info('Best validation loss: %f', valid_obj)
  logging.info('Best validation accuracy: %f', valid_acc)
  cell_encode = translate_genotype_to_encode(genotype)
  write_array_to_file(cell_encode, os.path.join(args.save, 'best_genotype.txt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch): #perturb_alpha, epsilon_alpha
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

        architect.step(input, target, input_search, target_search, lr, optimizer, epoch, unrolled=args.unrolled, betadecay=args.betadecay,
                       sam=args.sam)
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        # print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()

        '''
        # perturb on alpha
        # print('after softmax', model.arch_parameters())
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        # print('after perturb', model.arch_parameters())
        '''

        logits = model(input) #, updateType='weight')
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

