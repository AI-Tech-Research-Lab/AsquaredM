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

def get_limits_plot(dataset):
    if dataset == 'cifar10':
        return 91, 94
    elif dataset == 'cifar100':
        return 70, 78

def plot_histogram(data_array, bins=100, path='', baselines=None, dataset='cifar10'):

    FONT_SIZE = 8
    FIGSIZE = (4, 5)
    #COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]
    titles = ['DARTS', 'SAM']

    num_plots = len(data_array) 

    # Set up subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=FIGSIZE, sharex=True)  # same scale x-axis

    # Plot histograms and curves for each element in the array
    for i, data in enumerate(data_array):
        sns.histplot(data, bins=bins, color='darkblue', edgecolor='black', kde=True, line_kws={'linewidth': 1}, ax=axs[i], stat='density')
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)  # Hide y-axis values
        min_val, max_val = get_limits_plot(dataset)
        axs[i].set_ylim(0, 1.0)  # Set y-axis limits

        axs[i].set_title(f"{titles[i]} (Test accuracy: {baselines[i]:.2f}%)")  # Adjust title as needed

        # Plot baselines as vertical lines
        if baselines:
            axs[i].axvline(baselines[i], color='red', linestyle='--', linewidth=1)

        # Add grid with custom interval
        axs[i].grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
        axs[i].set_yticks(np.arange(0, 0.9, 0.2))

    # Apply common x-axis limits to all subplots
    for ax in axs:
        ax.set_xlim(min_val, max_val)

    # Add common X-axis label
    axs[-1].set_xlabel('Accuracy', fontsize=FONT_SIZE)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_neighbors(folder, dataset='cifar10', radius=1, baselines=None):
    models=['DARTS_seed3', 'SAM_exp1_seed7']
    accs=[0,0]
    folder1 = folder + '_arch' + models[0] + '_radius' + str(radius)
    filename=os.path.join(folder1,'archive_darts.txt')
    accs[0], _ = read_best_valid_acc_from_file(filename)
    folder2 = folder + '_arch' + models[1] + '_radius' + str(radius)
    filename=os.path.join(folder2,'archive_darts.txt')
    accs[1], _ = read_best_valid_acc_from_file(filename)
    plot_histogram(accs, bins=50, path=os.path.join(folder1,'histogram_darts_dataset' + dataset+'_radius' + str(radius) + '.pdf'), 
                    dataset=dataset, baselines=baselines)

# Usage
plot_neighbors('results/darts_train_neighbors_datasetcifar100', dataset='cifar100', radius=1, baselines=[73.5, 74.7])
'''
folder = 'results/darts_train_neighbors_datasetcifar100_archBETADARTS'
filename=os.path.join(folder,'archive_darts.txt')
accs = read_best_valid_acc_from_file(filename)
accs = accs[0] + [accs[1]]
plot_histogram(accs, bins=100, path=os.path.join(folder,'plot.png'), 
                dataset='cifar100', baseline=73.86)
'''
