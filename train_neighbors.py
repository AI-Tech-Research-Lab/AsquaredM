import argparse
from collections import OrderedDict
import glob
import subprocess
from nasbench201.archive import NASBench201
from optimizers.darts import utils
from sota.cnn.model_search import Network
import torch

import sys
import os
import json
import numpy as np
import logging

sys.path.append(os.getcwd())
 
from train_utils import get_dataset, get_optimizer, get_loss, get_lr_scheduler, get_data_loaders, load_checkpoint, validate, initialize_seed, \
                        Log, train, get_net_info
from perturb import get_net_info_runtime
#from nasbench201.nasbenchnet import NASBenchNet
from sota.cnn.darts import DARTS

def check_if_evaluated(config, output_path):
    config_str = ','.join(map(str, config))
    save_path = os.path.join(output_path, f"net_{config_str}.pt")
    if os.path.exists(save_path):
        return True, save_path
    return False, save_path

def call_training_script(genotype,id,args):
    # Prepare the experiment directory
    exp_dir = os.path.join(args.save, 'neighbor_' + str(id))

    bash_file = ['#!/bin/bash']
    cfg =OrderedDict()
    
    cfg['dataset'] = args.dataset
    cfg['data'] = args.data
    cfg['device'] = args.device
    cfg['output_path'] = args.save
    cfg['epochs'] = 600
    cfg['batch_size'] = 96
    cfg['momentum'] = 0.9
    cfg['drop_path_prob'] = 0.2
    cfg['auxiliary'] = True
    cfg['auxiliary_weight'] = 0.4
    cfg['cutout'] = True
    cfg['seed'] = args.seed

    execution_line = "python sota/cnn/train.py"
    for k, v in cfg.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    
    bash_file.append(execution_line)
    
    bash_file_path = os.path.join(args.save, 'run_bash.sh')
    with open(bash_file_path, 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)
    
    utils.create_exp_dir(exp_dir, scripts_to_save=None)

    subprocess.call("sh {}/run_bash.sh".format(args.save), shell=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/remote-home/source/share/dataset',
                    help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
    parser.add_argument('--save', type=str, default='/remote-home/source/share/dataset',
                    help='location of the exp folder')
    parser.add_argument('--arch', type=str, default='/remote-home/source/share/dataset',
                    help='initial config')
    parser.add_argument('--radius', type=int, default=1,
                    help='radius')
    parser.add_argument('--samples', type=int, default=1,
                    help='number of neighbors')
    args = parser.parse_args()

    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    darts = DARTS(2)
    genotype = eval("genotypes.%s" % args.arch)
    logging.info(genotype)

    matrix = darts.genotype_to_adjacency_matrix(genotype)
    radius=1
    samples=2
    neighbors = darts.sample_neighbors(matrix, args.radius, args.samples)
    results = {}

    for neighbor in neighbors:
        logging.info(f"Evaluating config: {neighbor}")
        already_evaluated, save_path = check_if_evaluated(neighbor, args.archive_path)
        
        if already_evaluated:
            logging.info(f"Configuration {neighbor} already evaluated. Loading results from {save_path}")
            state_dict = torch.load(save_path)
            top1_test = state_dict['accuracy']
        else:
            #launch process
            torch.save({'accuracy': top1_test, 'state_dict': state_dict}, save_path)
        
        config_str = ','.join(map(str, neighbor))
        results[config_str] = {
            'accuracy': top1_test,
            'state_dict': state_dict
        }
        
        save_path = os.path.join(args.output_path, f"net_{config_str}.pt")
        torch.save(state_dict, save_path)

    with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    logging.info("All configurations evaluated and results saved.")