from itertools import product
import os
from nasbench201.archive import NASBench201
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

def neighbors_by_radius(N, possible_values, vector, radius):
    neighbors = []
    for config in product(possible_values, repeat=N):
        diff_count = sum([1 for x, y in zip(vector, config) if x != y])
        if diff_count == radius:
            neighbors.append(config)
    return neighbors

def avg_val_acc(bench, configs):
    avg_val_acc = 0
    accs = []
    for c in configs:
        arch = bench.decode(c)
        val_acc = bench.get_info_from_arch(arch)['val-acc']
        accs.append(val_acc)
        avg_val_acc += val_acc
    return avg_val_acc/len(configs), accs, np.std(accs)

def rank_by_val_acc(bench):
    if bench.dataset=='cifar10':
        val_dataset = bench.dataset + '-valid'
    else:
        val_dataset = bench.dataset
    val_accs = bench.archive['val-acc'][val_dataset]
    idxs = list(range(len(val_accs)))
    val_accs_idxs = list(zip(val_accs, idxs))
    sorted_val_accs_idxs = sorted(val_accs_idxs, key=lambda x: x[0])
    sorted_val_accs = [val_acc for val_acc, _ in sorted_val_accs_idxs]
    sorted_idxs = [idx for _, idx in sorted_val_accs_idxs]
    # Filter the indices for architectures with specific validation accuracies
    filtered_idxs = []
    filtered_idxs.append(sorted_idxs[-1])  # Add the best architecture
    # cifar10: 80,70 cifar100: 65,55 imagenet: 35,25
    if bench.dataset == 'cifar10':
        max = 80
        min = 70
    elif bench.dataset == 'cifar100':
        max = 65
        min = 55
    else: # imagenet
        max = 35
        min = 25
    found_max= False 
    found_min = False
    for val_acc, idx in reversed(sorted_val_accs_idxs):
        if not found_max and val_acc < max:
            filtered_idxs.append(idx)
            found_max = True
        
        if not found_min and val_acc < min:
            filtered_idxs.append(idx)
            found_min = True
        
        if found_max and found_min:
            break  # Stop if both conditions are met
    
    filtered_val_accs = [val_acc for val_acc, _ in sorted_val_accs_idxs if _ in filtered_idxs]

    return filtered_val_accs, filtered_idxs

def pvalue(data):
    import scipy.stats as stats

    # Population mean
    population_mean = 0.5
    
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(data, population_mean)
    
    print("t-statistic:", t_stat)
    print("p-value:", p_value)
    
    # Conditions
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis. There is enough evidence to suggest a significant difference.")
    else:
        print("Fail to reject the null hypothesis. The difference is not statistically significant.")

def plot_avgacc_vs_radius():

    bench = NASBench201(dataset='cifar10')
    sorted_val_accs, sorted_idxs = rank_by_val_acc(bench)
    print("SORTED VAL ACCS: ", sorted_val_accs)

    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
    config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
    radius_range=range(1,4)
    # Placeholder lists to store accuracy values
    acc_config1 = [sorted_val_accs[2]]
    acc_config2 = [sorted_val_accs[1]]
    acc_config3 = [sorted_val_accs[0]]

    for radius in radius_range:
        # Calculate neighbors for each configuration
        neighbors_config1 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, radius)
        neighbors_config2 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config2, radius)
        neighbors_config3 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config3, radius)
        
        # Calculate average validation accuracy for each configuration
        avg_acc_config1 = avg_val_acc(bench, neighbors_config1)[0]
        avg_acc_config2 = avg_val_acc(bench, neighbors_config2)[0]
        avg_acc_config3 = avg_val_acc(bench, neighbors_config3)[0]
        
        # Append accuracy values to respective lists
        acc_config1.append(avg_acc_config1)
        acc_config2.append(avg_acc_config2)
        acc_config3.append(avg_acc_config3)

    # Plotting
    radius_range = range(4)
    plt.plot(radius_range, acc_config1, marker='s', label='Config 1')
    plt.plot(radius_range, acc_config2, marker='s', label='Config 2')
    plt.plot(radius_range, acc_config3, marker='s', label='Config 3')
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Accuracy vs. Radius for Different Configurations')
    plt.legend()
    plt.savefig('accuracy_vs_radius.png')
    plt.show()

def boxplot_acc_vs_radius():

    bench = NASBench201(dataset='cifar10')
    sorted_val_accs, sorted_idxs = rank_by_val_acc(bench)
    print("SORTED VAL ACCS: ", sorted_val_accs)

    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
    config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
    
    radius_range=range(1,4)
    # Placeholder lists to store accuracy values
    accuracies_by_radius_config1 = []
    accuracies_by_radius_config2 = []
    accuracies_by_radius_config3 = []

    for radius in radius_range:
        # Calculate neighbors for each configuration
        neighbors_config1 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, radius)
        neighbors_config2 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config2, radius)
        neighbors_config3 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config3, radius)
        
        # Calculate validation accuracy for each configuration
        acc_neighbors_config1 = avg_val_acc(bench, neighbors_config1)[1]
        acc_neighbors_config2 = avg_val_acc(bench, neighbors_config2)[1]
        acc_neighbors_config3 = avg_val_acc(bench, neighbors_config3)[1]
        
        # Append accuracy values to respective lists
        accuracies_by_radius_config1.append(acc_neighbors_config1)
        accuracies_by_radius_config2.append(acc_neighbors_config2)
        accuracies_by_radius_config3.append(acc_neighbors_config3)

    # Plotting the boxplots
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.boxplot(accuracies_by_radius_config1, patch_artist=True)
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Config 1')

    plt.subplot(1, 3, 2)
    plt.boxplot(accuracies_by_radius_config2, patch_artist=True)
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Config 2')

    plt.subplot(1, 3, 3)
    plt.boxplot(accuracies_by_radius_config3, patch_artist=True)
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Config 3')

    # Adding grid
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Showing plot
    plt.savefig('accuracy_vs_radius_boxplot.png')
    plt.show()

def plot_histograms(data_array, bins=100, path='', baselines=None):

    FONT_SIZE = 8
    #FIGSIZE = (3.5, 3.0)
    FIGSIZE = (4,5)
    COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]

    num_plots = len(data_array) - 1 

    # Set up subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=FIGSIZE, sharex=False)

    all_dataset = data_array[0]

    # Plot histograms and curves for each element in the array
    for i, data in enumerate(data_array[1:]):
        data = np.array(data)   
        data = data[data > 10]
        # Plot transparent curve behind histogram for the first element
        sns.histplot(all_dataset, bins=bins, color='green', edgecolor='black', kde=True, line_kws={'linewidth': 1, 'alpha': 0.2}, ax=axs[i], stat='density')
        sns.histplot(data, bins=bins, color='darkblue', edgecolor='black', kde=True, line_kws={'linewidth': 1}, ax=axs[i], stat='density')
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)  # Hide y-axis values
        #axs[i].set_xlim(60, 95)  # Set x-axis limits cifar10
        #axs[i].set_xlim(40, 75) #cifar100
        axs[i].set_xlim(10, 50)
        axs[i].set_ylim(0, 0.8)  # Set y-axis limits
        # Add title to each subplot
        axs[i].set_title(f'Plot {i+1}')  # Adjust title as needed
        # Plot baselines as vertical lines
        if baselines:
                axs[i].axvline(baselines[i], color='red', linestyle='--', linewidth=1)
        
        # Add grid with custom interval
        axs[i].grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
        axs[i].set_yticks(np.arange(0, 0.9, 0.2))

    # Add common X-axis label
    axs[-1].set_xlabel('Value', fontsize=FONT_SIZE)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(path, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_simple_histograms(data_array, bins=36, path=''):
    FONT_SIZE = 8
    FIGSIZE = (3.5, 3.0)
    COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]

    num_plots = len(data_array)

    # Set up subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=FIGSIZE, sharex=False)

    # Ensure axs is always a list
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # Plot histograms and curves for each element in the array
    for i, data in enumerate(data_array):
        sns.histplot(data, bins=bins, color=COLORS[i % len(COLORS)], edgecolor='black', kde=True, line_kws={'linewidth': 1}, ax=axs[i])
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis values
        axs[i].set_xlabel('Value', fontsize=FONT_SIZE)
        axs[i].set_xlim(10, 50)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the plot to a file
    if path:
        plt.savefig(path, bbox_inches='tight')

    # Show the plot
    plt.show()

def compute_acc_by_radius(dataset='cifar10'):
    bench = NASBench201(dataset=dataset)
    result_dir = os.path.join('../results/flatness_exp', dataset)  
    sorted_val_accs, sorted_idxs = rank_by_val_acc(bench)
    val_accs = bench.archive['val-acc'][dataset]

    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
    config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})

    configs = [config1, config2, config3]

    radius_range=range(1,4)
    #radius_range=range(1,2)
    # Placeholder lists to store accuracy values
    accuracies_by_radius_configs = []

    for idx in range(len(configs)):

        acc_neighbors_configs = []
        for radius in radius_range:
            model_path = os.path.join(result_dir,"accuracies_config_" + str(idx+1) + "_radius_" + str(radius) + ".npy")
            if not os.path.exists(model_path):
                print("Calculating array")
                config=configs[idx]
                # Calculate neighbors for each configuration
                neighbors_config = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config, radius)
                
                # Calculate validation accuracy for each configuration
                acc_neighbors_config = avg_val_acc(bench, neighbors_config)[1]
                
                acc_neighbors_config = np.array(acc_neighbors_config)

                # Save the numpy arrays to a single file
                np.save(model_path, acc_neighbors_config)

            else:
                print("Loading array")
                acc_neighbors_config = np.load(model_path)
                acc_neighbors_configs.append(acc_neighbors_config)
                #print("ACC NEIGHBORS CONFIG: ", acc_neighbors_config.shape)
        
        accuracies_by_radius_configs.append(acc_neighbors_configs)
    
    #give the config of neighbor of radius 3 of the first config with best val acc

    #idx = np.argmax(accuracies_by_radius_configs[0][2])
    #config = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, 3)[idx]

    acc_by_configs = [val_accs, accuracies_by_radius_configs[0], accuracies_by_radius_configs[1], accuracies_by_radius_configs[2]]
    return acc_by_configs
    #plot_histograms(acc_by_configs, path=os.path.join(result_dir,'histogram_configs.png'), baselines=(sorted_val_accs)[::-1]) 

def search_tree(root_config, target_config):
        def generate_moves(curr_config, target_config, path):
            if curr_config == target_config:
                # Base case: We have reached the target configuration
                return [path + [curr_config]]
            
            # Initialize a list to store all possible moves
            possible_moves = []

            #idxs of different integers between configs
            idxs = [i for i in range(len(curr_config)) if curr_config[i] != target_config[i]]

            for i in idxs:
                    # Generate a new configuration by incrementing the i-th integer
                    new_config = curr_config.copy()
                    new_config[i] = target_config[i]
                    new_paths = generate_moves(new_config, target_config, path + [curr_config])
                    possible_moves.extend(new_paths)
            
            return possible_moves
        return generate_moves(root_config, target_config, [])

def remove_outliers(data):
    from scipy import stats

    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(data))

    # Define a threshold for outlier detection (e.g., Z-score greater than 3)
    threshold = 3

    # Get indices of outliers
    outlier_indices = np.where(z_scores > threshold)[0]

    # Remove outliers
    new_data = np.delete(data, outlier_indices)
    new_indices = np.delete(np.arange(len(data)), outlier_indices)
    return new_data, new_indices

def path_bench(dataset):
    print("DATASET: ", dataset)
    bench = NASBench201(dataset=dataset)
    #net1 config
    sorted_val_accs, sorted_idxs = rank_by_val_acc(bench)
    if dataset=='cifar10':
        config_idx = 0
    elif dataset=='cifar100':
        config_idx = 1
    else:
        config_idx = 2
    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    print("CONFIG1: ", config1)
    arch = bench.decode(config1)
    acc1 = bench.get_info_from_arch(arch)
    #net2 config: give the config of neighbor of radius 3 of the first config with best val acc
    acc_radius3= compute_acc_by_radius(dataset)[config_idx+1][2]
    new_data,new_indices = remove_outliers(acc_radius3)
    idx = new_indices[np.argmax(new_data)]
    config2 = list(neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, 3)[idx])
    arch=bench.decode(config2)
    acc2 = bench.get_info_from_arch(arch)
    print("CONFIG2: ", config2)
    # Find the paths between the two configurations
    paths = search_tree(config1, config2)
    #print("PATHS: ", paths)
    #print(len(paths))
    # Average by level of the tree
    max_level = len(paths[0])-2
    archs_by_level = [[] for _ in range(max_level)]
    for i in range(max_level):
        for path in paths:
                archs_by_level[i].append(path[i+1])
    #print("CONFIGS BY LEVEL: ", archs_by_level)
    # Average validation accuracy by level
    avg_val_accs = []
    std_val_accs = []
    for configs in archs_by_level:
        avg, _, std = avg_val_acc(bench, configs)
        avg_val_accs.append(avg)
        std_val_accs.append(std)
    path_accs= [acc1['val-acc']] + avg_val_accs + [acc2['val-acc']]
    print("PATH ACCS: ", path_accs)
    print("STD VAL ACCS: ", std_val_accs)
    # Plot the path
    # (flat plot is better)
    # Plot the path
    x = list(range(4))  # 0, 1, 2, 3 representing the radius
    y = path_accs
    yerr = [0] + std_val_accs + [0]  # no std dev for acc1 and acc2
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, capthick=2, elinewidth=1, label='Validation Accuracy')
    plt.xlabel('Radius')
    plt.ylabel('Accuracy')
    plt.title(f'Path Accuracies for {dataset}')
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.show()

    if not os.path.exists('results/plots'):
        os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/path_accs_'+dataset+'.png')

path_bench('cifar10')
path_bench('cifar100')
path_bench('ImageNet16-120')