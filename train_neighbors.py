import argparse
from collections import OrderedDict
import glob
import subprocess
import torch
import json
import numpy as np
import logging
import os
import sys

sys.path.append(os.getcwd())
from sota.cnn.darts import DARTS
from sota.cnn import genotypes
from sota.cnn.genotypes import Genotype

def check_if_evaluated(config, output_path):
    """
    Checks if the configuration has been evaluated by searching in the .txt file.
    
    Parameters:
    - config (tuple or list): Configuration parameters to be checked.
    - output_path (str): Path where the .txt file is saved.
    
    Returns:
    - (bool, tuple): A tuple where the first element is a boolean indicating 
      whether the configuration exists in the .txt file, and the second element 
      is a tuple (config, stats) or (config, None) if not found.
    """
    config_str = ','.join(map(str, config))
    archive_path = os.path.join(output_path, 'archive_darts.txt')
    
    if not os.path.exists(archive_path):
        return False, (config, None)
    
    try:
        with open(archive_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return False, (config, None)
    
    for line in lines:
        entry_config_str, stats = line.strip().split(':', 1)
        if entry_config_str == config_str:
            return True, (config, json.loads(stats))
    
    return False, (config, None)

def save_results(config, stats, output_path):
    """
    Save results to a text file.
    
    Parameters:
    - config (tuple or list): Configuration parameters.
    - stats (dict): Metrics to be saved.
    - output_path (str): Path where the .txt file is saved.
    """
    config_str = ','.join(map(str, config))
    archive_path = os.path.join(output_path, 'archive_darts.txt')
    
    try:
        with open(archive_path, 'a') as file:
            file.write(f"{config_str}:{json.dumps(stats)}\n")
    except Exception as e:
        print(f"Error writing file: {e}")

def call_training_script(genotypes, args):
    """
    Create and execute a bash script to train multiple genotypes on different GPUs.
    
    Parameters:
    - genotypes (list of dict): List of genotype configurations.
    - args (argparse.Namespace): Command-line arguments.
    """
    bash_file_path = os.path.join(args.save, 'run_bash.sh')
    with open(bash_file_path, 'w') as handle:
        handle.write('#!/bin/bash\n')

        for id, genotype in enumerate(genotypes):
            exp_dir = os.path.join(args.save, f'neighbor_{id}')
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)

            genotype['normal_concat'] = list(genotype['normal_concat'])
            genotype['reduce_concat'] = list(genotype['reduce_concat'])

            with open(os.path.join(exp_dir, 'genotype.json'), 'w') as f:
                json.dump(genotype, f)

            # Determine the GPU to use
            gpu_id = args.gpus[id % len(args.gpus)]

            cfg = OrderedDict()
            cfg['gpu'] = gpu_id
            cfg['dataset'] = args.dataset
            cfg['data'] = args.data
            cfg['save'] = exp_dir
            cfg['epochs'] = args.epochs
            cfg['train_limit'] = args.train_limit
            cfg['batch_size'] = 96
            cfg['momentum'] = 0.9
            cfg['drop_path_prob'] = 0.2
            cfg['auxiliary'] = True
            cfg['auxiliary_weight'] = 0.4
            cfg['cutout'] = True
            cfg['seed'] = args.seed

            execution_line = f"CUDA_VISIBLE_DEVICES={gpu_id} python sota/cnn/train.py"
            for k, v in cfg.items():
                if v is not None:
                    if isinstance(v, bool):
                        if v:
                            execution_line += f" --{k}"
                    else:
                        execution_line += f" --{k} {v}"

            handle.write(f"{execution_line}\n")

    # Execute the bash script
    subprocess.call(f"sh {bash_file_path}", shell=True)


def update_archive_from_stats(exp_dir, archive_path):
    """
    Update the archive file by reading stats from each experiment directory.
    
    Parameters:
    - exp_dir (str): The base directory where experiment folders are located.
    - archive_path (str): Path to the archive file.
    """
    results = []

    # Iterate over all experiment directories
    for folder in os.listdir(exp_dir):
        folder_path = os.path.join(exp_dir, folder)
        if os.path.isdir(folder_path):
            stats_path = os.path.join(folder_path, 'stats.json')
            print("STATS PATH:", stats_path)
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                        
                        # Extract and remove 'config' from stats
                        config = stats.pop('config', None)
                        print("Config:", config)
                        
                        if config is not None:
                            # Convert config to a JSON string
                            config_str = json.dumps(config)
                            # Collect the results as a tuple (config_str, stats)
                            results.append((config_str, stats))
                except Exception as e:
                    print(f"Error reading stats file {stats_path}: {e}")

    # Write results to the archive file
    with open(archive_path, 'w') as f:
        for config_str, stats in results:
            f.write(f"{config_str}:{json.dumps(stats)}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/remote-home/source/share/dataset',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
    parser.add_argument('--save', type=str, default='/remote-home/source/share/dataset',
                        help='location of the exp folder')
    parser.add_argument('--arch', type=str, default='GenotypeName',
                        help='initial config')
    parser.add_argument('--radius', type=int, default=1,
                        help='radius')
    parser.add_argument('--samples', type=int, default=1,
                        help='number of neighbors')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='list of gpu device ids')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--train_limit', type=float, default=0.0, help='training loss limit')
    args = parser.parse_args()

    # Ensure the exp directory exists
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Define the path to the .txt file
    archive_path = os.path.join(args.save, 'archive_darts.txt')
    if not os.path.exists(archive_path):
        with open(archive_path, 'w') as f:
            pass  # Just create an empty file

    darts = DARTS(2)
    genotype = eval(f"genotypes.{args.arch}")
    dict_ = darts.to_dict(genotype)
    matrix = darts.genotype_to_adjacency_matrix(dict_)
    neighbors = darts.sample_neighbors(matrix, args.radius, args.samples)

    print("Neighbors:")
    print(neighbors)

    #for id, neighbor in enumerate(neighbors):
    #logging.info(f"Evaluating config: {neighbor}")

        #already_evaluated, _ = check_if_evaluated(neighbor, args.save)
        
        #if already_evaluated:
        #logging.info(f"Configuration {neighbor} already evaluated. Loading results from {archive_path}.")
        #else:
        # Use each GPU for training

    call_training_script(neighbors, args)

    # Update the archive file after all trainings are complete
    update_archive_from_stats(args.save, archive_path)

    logging.info("All configurations evaluated and results saved.")
