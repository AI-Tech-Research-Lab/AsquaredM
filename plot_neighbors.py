import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

def read_best_valid_acc_from_file(filename):
    
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

def plot_histogram(data, bins=100, path='', baseline=None, dataset='cifar10'):
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
    ax.set_xlabel('Value', fontsize=FONT_SIZE)
    ax.set_title('Histogram', fontsize=FONT_SIZE)
    
    # Add grid and customize y-axis ticks
    ax.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax.set_yticks(np.arange(0, 0.9, 0.2))

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(path, bbox_inches='tight')
    plt.show()

# Usage
folder = 'results/darts_train_neighbors_datasetcifar100_archBETADARTS'
filename=os.path.join(folder,'archive_darts.txt')
accs = read_best_valid_acc_from_file(filename)
accs = accs[0] + [accs[1]]
plot_histogram(accs, bins=100, path=os.path.join(folder,'plot.png'), 
                dataset='cifar100', baseline=73.86)
