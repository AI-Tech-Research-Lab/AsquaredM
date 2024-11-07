import argparse
from collections import OrderedDict
import glob
import math
import subprocess
import torch
import json
import numpy as np
import logging
import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

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

def read_val_accs_from_archive(filename):
    
    array=[]
    baseline = True
    acc_baseline=0
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Make sure the line is not empty

                # Split the line into the architecture and metrics part
                arch_part, metrics_part = line.split('}:')

                # Parse the architecture part
                architecture_json = arch_part + '}'
                try:
                    architecture = json.loads(architecture_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode architecture JSON: {e}")
                    continue

                # Parse the metrics part
                metrics_json = metrics_part.strip()
                try:
                    metrics = json.loads(metrics_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode metrics JSON: {e}")
                    continue

                valid_acc = metrics.get("best_valid_acc", -float('inf'))

                if baseline: #the first line refers to the baseline
                    acc_baseline = valid_acc
                    baseline = False
                else:
                    array.append(valid_acc)

    return array, acc_baseline

'''
def read_val_accs_path_from_archive(filename,n):
    
    array=[]
    avg_array=[]
    baseline = True
    acc_baseline=0
    K=1
    to_samples = 3 #math.comb(n,K) #number of samples to sample for level K (binomial coefficient)

    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Make sure the line is not empty

                # Split the line into the architecture and metrics part
                arch_part, metrics_part = line.split('}:')

                # Parse the architecture part
                architecture_json = arch_part + '}'
                try:
                    architecture = json.loads(architecture_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode architecture JSON: {e}")
                    continue

                # Parse the metrics part
                metrics_json = metrics_part.strip()
                try:
                    metrics = json.loads(metrics_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode metrics JSON: {e}")
                    continue

                valid_acc = metrics.get("best_valid_acc", -float('inf'))

                if baseline: #the first line refers to the baseline
                    acc_baseline = valid_acc
                    baseline = False
                else:
                    if K == n: #last sample
                        acc_target=valid_acc
                    else:
                        to_samples -= 1
                        array.append(valid_acc)
                        if to_samples == 0:
                            #average the samples of array, append the value to avg_array, and clean the array, increase the level
                            avg_array.append(sum(array)/len(array))
                            stds = np.std(array)
                            array.clear()
                            K+=1
                            to_samples = 3 #math.comb(n,K)
    
    avg_array.insert(0, acc_baseline)
    avg_array.append(acc_target)
    return avg_array, stds
'''

def read_val_accs_path_from_archive(filename, n):
    """
    Reads validation accuracies from an archive and averages samples for different levels,
    calculating standard deviations for each level.
    """
    array = []
    avg_array = []
    std_array = []
    K = 1
    to_samples = 3  # Number of samples per level

    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                # Split the line into the architecture and metrics part
                arch_part, metrics_part = line.split('}:')

                # Parse the architecture part
                architecture_json = arch_part + '}'
                try:
                    architecture = json.loads(architecture_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode architecture JSON: {e}")
                    continue

                # Parse the metrics part
                metrics_json = metrics_part.strip()
                try:
                    metrics = json.loads(metrics_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode metrics JSON: {e}")
                    continue

                valid_acc = metrics.get("best_valid_acc", -float('inf'))

                to_samples -= 1
                array.append(valid_acc)
                if to_samples == 0:
                    # Average the samples in the array, append to avg_array, and calculate std deviation
                    avg_array.append(sum(array) / len(array))
                    std_array.append(np.std(array))
                    array.clear()
                    K += 1
                    to_samples = 3  # Reset the number of samples per level

    return avg_array, std_array

def plot_histogram(data, bins=100, path='', baseline=None, dataset='cifar10', radius=1):
    FONT_SIZE = 8
    FIGSIZE = (8, 4)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot histogram and KDE
    sns.histplot(data, bins=bins, color='darkblue', edgecolor='black', kde=True, line_kws={'linewidth': 1}, ax=ax, stat='density')
    
    # Add vertical line for baseline if provided
    if baseline is not None:
        ax.axvline(baseline, color='red', linestyle='--', linewidth=1)
    
    # Set axis labels and title
    ax.set_xlabel('Accuracy', fontsize=FONT_SIZE)
    ax.set_title('Locality in the neighborhood of radius '+ str(radius) + ' on ' + dataset, fontsize=FONT_SIZE)
    
    # Add grid and customize y-axis ticks
    ax.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax.set_yticks(np.arange(0, 0.9, 0.2))

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_line(path_accs, std_test_accs, output_path, dataset):
    """
    Plot a line graph of test accuracies.

    Args:
    - accuracies (list): List of test accuracies to plot.
    - output_path (str): Path to save the plot image.
    """
    x = list(range(4))  # 0, 1, 2, 3 representing the radius
    y = path_accs
    yerr = [0] + std_test_accs + [0]  # no std dev for acc1 and acc2
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, capthick=2, elinewidth=1, label='Test Accuracy')
    plt.xlabel('Radius')
    plt.ylabel('Accuracy')
    plt.title(f'Path Accuracies for {dataset}')
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(output_path + '/path_accs.pdf', format='pdf', bbox_inches='tight', dpi=300)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/remote-home/source/share/dataset',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
    parser.add_argument('--save', type=str, default='/remote-home/source/share/dataset',
                        help='location of the exp folder')
    parser.add_argument('--arch', type=str, default='GenotypeName',
                        help='initial config')
    parser.add_argument('--arch_target', type=str, default=None,
                        help='target config for path')
    parser.add_argument('--acc_ref', type=float, default=0.0,
                        help='reference accuracy for the baseline')
    parser.add_argument('--acc_target', type=float, default=0.0,
                        help='reference accuracy for the baseline')
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

    if args.arch_target:
        genotype_target = eval(f"genotypes.{args.arch_target}")
        dict_target = darts.to_dict(genotype_target)
        #matrix_target = darts.genotype_to_adjacency_matrix(dict_target)
        num_differences=3
        neighbors = darts.sample_neighbors_path(dict_, dict_target, num_differences-1)
        # flatten the list
        #neighbors = [item for sublist in neighbors for item in sublist]
        #add target genotype to end of the list
        #neighbors.append(dict_target)
    else:
        neighbors = darts.sample_neighbors(matrix, args.radius, args.samples)
        # add baseline genotype at the top of the list
        neighbors.insert(0, dict_) 

    print("Len Neighbors:")
    print(len(neighbors))

    call_training_script(neighbors, args)

    # Update the archive file after all trainings are complete
    update_archive_from_stats(args.save, archive_path)

    # Read the validation accuracies from the archive file and plot the histogram

    if args.arch_target:
        accs, stds = read_val_accs_path_from_archive(archive_path, num_differences)
        accs.insert(0, args.acc_ref)
        accs.append(args.acc_target)
        plot_line(accs, stds, args.save, args.dataset)
    else:
        accs, baseline = read_val_accs_from_archive(archive_path)
        if 'DARTS' in args.arch:
            model = 'darts'
        else:
            model = 'sam'
        filename = 'histogram_' + model + '_neighbors_' + args.dataset + '_radius' + str(args.radius) + '.pdf'
        plot_histogram(accs, path=os.path.join(args.save, filename), baseline=baseline, dataset=args.dataset, radius=args.radius)

    logging.info("All configurations evaluated and results saved.")
