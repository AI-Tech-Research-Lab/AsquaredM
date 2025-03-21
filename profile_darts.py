import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import math

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
        return 91, 95
    elif dataset == 'cifar100':
        return 70, 78

def compute_barrier(accs, acc_min):
    acc_a = accs[0]
    acc_b = accs[1]
    #acc_min = min(accs)
    return np.round(0.5 * (acc_a + acc_b) - acc_min,2)

def read_val_accs_path_from_archive(filename, n):
    """
    Reads validation accuracies from an archive and averages samples for different levels,
    calculating standard deviations for each level.
    """
    array = []
    avg_array = []
    std_array = []
    accs=[]
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
                accs.append(valid_acc)
                if to_samples == 0:
                    # Average the samples in the array, append to avg_array, and calculate std deviation
                    avg_array.append(sum(array) / len(array))
                    std_array.append(np.std(array))
                    array.clear()
                    K += 1
                    to_samples = 3  # Reset the number of samples per level

    return avg_array, std_array, accs

def get_bins(dataset, radius):

    if dataset == 'cifar10':
        if radius == 1 :
            return 100, 100
        elif radius == 2:
            return 100, 100
        elif radius == 3:
            return 100, 100
    elif dataset == 'cifar100':
        if radius == 1:
            return 100, 100
        elif radius == 2:
            return 100, 100
        elif radius == 3:
            return 100, 100

def plot_histogram(data_array, bins=100, path='', baselines=None, dataset='cifar10', radius=1):

    FONT_SIZE = 8
    FIGSIZE = (4, 5)
    #COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]
    titles = ['DARTS', 'SAM']

    num_plots = len(data_array) 

    # Set up subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=FIGSIZE, sharex=True)  # same scale x-axis

    # Plot histograms and curves for each element in the array
    for i, data in enumerate(data_array):
        temp_bins = get_bins(dataset, radius)[i]
        sns.histplot(data, bins=bins, color='darkblue', edgecolor='black', kde=True, line_kws={'linewidth': 2}, ax=axs[i], stat='density')
        axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)  # Show x-axis labels for all subplots
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)  # Hide y-axis values
        min_val, max_val = get_limits_plot(dataset)
        
        # Set y-axis limits
        y_values = [bar.get_height() for bar in axs[i].patches]
        max_density = max(y_values) 
        #max_density = max_density if max_density < 0.8 else 0.8 # Limit the maximum density to 0.8
        ymax = math.ceil(max_density * 10) / 10
        axs[i].set_ylim(0, ymax)  

        axs[i].set_title(f"{titles[i]} (Test accuracy: {baselines[i]:.2f}%)")  # Adjust title as needed

        # Plot baselines as vertical lines
        if baselines:
            axs[i].axvline(baselines[i], color='red', linestyle='--', linewidth=1)

        # Add grid with custom interval
        axs[i].grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
        axs[i].set_yticks(np.arange(0, ymax, 0.2))

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
    plot_histogram(accs, bins=10, path=os.path.join(folder1,'histogram_darts_dataset' + dataset+'_radius' + str(radius) + '.pdf'), 
                    dataset=dataset, baselines=baselines, radius=radius)

def path_bench_qualities(paths, dataset):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    FONT_SIZE=20

    # Set font sizes
    plt.rcParams.update({
        'font.size': FONT_SIZE+4,         # General font size
        'axes.titlesize': FONT_SIZE+4,    # Title font size
        'axes.labelsize': FONT_SIZE+4,    # Axis label font size
        'xtick.labelsize': FONT_SIZE,   # X-tick label font size
        'ytick.labelsize': FONT_SIZE,   # Y-tick label font size
        'legend.fontsize': FONT_SIZE+2,   # Legend font size
    })

    #acc_base = {'DARTScifar10': 91.92, 'DARTScifar100': 73.5, 'SAMcifar10': 92.77, 'SAMcifar100': 74.7}
    #acc_target = {'DARTScifar10': 91.73, 'DARTScifar100': 73.54, 'SAMcifar10': 92.75, 'SAMcifar100': 74.78}

    acc_base = {'DARTScifar10': 96.91, 'DARTScifar100': 81.52, 'SAMcifar10': 97.4, 'SAMcifar100': 83.2, 'BADcifar10': 95.63, 'BADcifar100': 77.32}
    acc_target = {'DARTScifar10': 96.77, 'DARTScifar100': 81.33, 'SAMcifar10': 97.32, 'SAMcifar100': 83.16, 'BADcifar10': 95.7, 'BADcifar100': 77.45}


    # Define qualities
    qualities = ["DARTS", "SAM"]#, "BAD"]
    #paths = [path1, path2]
    barriers = []
    paths_by_quality = {}
    accs_by_quality = {}
    std_by_quality = {}

    # Process paths and compute barriers
    for i, quality in enumerate(qualities):
        avg_test_accs, std_test_accs, accs = read_val_accs_path_from_archive(paths[i], 3)
        acc1 = acc_base[quality+dataset]
        acc2 = acc_target[quality+dataset]
        
        # Store path accuracies and standard deviations for this quality
        path_accs = [acc1] + avg_test_accs + [acc2]
        paths_by_quality[quality] = path_accs
        std_by_quality[quality] = [0] + std_test_accs + [0]  # no std dev for acc1 and acc2
        accs_by_quality[quality] = (acc1, acc2)
        barriers.append(compute_barrier(path_accs, min(accs)))
        print(f"PATH ACCS for {quality}: ", path_accs)
        print(f"STD VAL ACCS for {quality}: ", std_test_accs)
    print("BARRIERS: ", barriers)

    # Plot the paths for qualities
    plt.figure(figsize=(10, 5))
    x = list(range(4))  # 0, 1, 2, 3 representing the radius
    colors = ['red', 'blue', 'orange']

    for i, quality in enumerate(qualities):
        y = paths_by_quality[quality]
        yerr = std_by_quality[quality]
        plt.errorbar(
            x, y, yerr=yerr, fmt='o-', color=colors[i], capsize=5, capthick=2, elinewidth=1, label=f'{quality}'
        )
        # Add squares for extreme points
        plt.scatter([x[0], x[-1]], [y[0], y[-1]], color=colors[i], s=80, marker='s', zorder=3)  # Squares
        # Add circles for internal points
        plt.scatter(x[1:-1], y[1:-1], color=colors[i], s=80, marker='o', zorder=3)  # Circles

    # Customize plot
    plt.xlabel('Radius')
    plt.ylabel('Accuracy')
    name = 'CIFAR-10' if dataset == 'cifar10' else 'CIFAR-100'
    plt.title(f'Path Accuracies for {name}')
    plt.xticks(x)
    plt.legend()
    plt.grid(False)

    # Save the plot to the results folder
    results_dir = f'results/flatness_exp_{dataset}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    plot_path = f'{results_dir}/path_accs_darts_'+dataset+'.pdf'
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # Save barriers in a JSON file
    barriers_path = f'{results_dir}/barriers.json'
    with open(barriers_path, 'w') as file:
        json.dump(barriers, file)

# Usage
'''
plot_neighbors('results/darts_train_neighbors_datasetcifar100', dataset='cifar100', radius=1, baselines=[73.5, 74.7])
plot_neighbors('results/darts_train_neighbors_datasetcifar100', dataset='cifar100', radius=2, baselines=[73.5, 74.7])
plot_neighbors('results/darts_train_neighbors_datasetcifar100', dataset='cifar100', radius=3, baselines=[73.5, 74.7])
plot_neighbors('results/darts_train_neighbors_datasetcifar10', dataset='cifar10', radius=1, baselines=[91.92, 92.77])
plot_neighbors('results/darts_train_neighbors_datasetcifar10', dataset='cifar10', radius=2, baselines=[91.92, 92.77])
plot_neighbors('results/darts_train_neighbors_datasetcifar10', dataset='cifar10', radius=3, baselines=[91.92, 92.77])
'''
paths=['results/darts_path_neighbors_datasetcifar10_archDARTS_seed3_arch_targetDARTS_TARGET_CIFAR10_radius3/archive_darts.txt', 
       'results/darts_path_neighbors_datasetcifar10_archSAM_exp1_seed7_arch_targetSAM_TARGET_CIFAR10_radius3/archive_darts.txt',
       'results/darts_path_neighbors_datasetcifar10_archSAM_exp_bad_seed1_arch_targetSAM_exp_bad2_seed1_radius3/archive_darts.txt']
path_bench_qualities(paths, 'cifar10')
paths=['results/darts_path_neighbors_datasetcifar100_archDARTS_seed3_arch_targetDARTS_TARGET_CIFAR100_radius3/archive_darts.txt', 
'results/darts_path_neighbors_datasetcifar100_archSAM_exp1_seed7_arch_targetSAM_TARGET_CIFAR100_radius3/archive_darts.txt', 
'results/darts_path_neighbors_datasetcifar100_archSAM_exp_bad_seed1_arch_targetSAM_exp_bad2_seed1_radius3/archive_darts.txt']

path_bench_qualities(paths, 'cifar100')

'''
folder = 'results/darts_train_neighbors_datasetcifar100_archBETADARTS'
filename=os.path.join(folder,'archive_darts.txt')
accs = read_best_valid_acc_from_file(filename)
accs = accs[0] + [accs[1]]
plot_histogram(accs, bins=100, path=os.path.join(folder,'plot.png'), 
                dataset='cifar100', baseline=73.86)
'''
