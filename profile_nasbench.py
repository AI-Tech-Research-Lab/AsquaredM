from itertools import product
import os
from nasbench201.archive import NASBench201
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import math

def find_neighbors_with_similar_performance(bench, accuracy_min, accuracy_max, radius=3, tolerance=1):
    val_dataset = bench.dataset
    test_accs = bench.archive['test-acc'][val_dataset]
    #architectures = bench.archive['architectures']

    # Filter architectures within the given accuracy range
    architectures_in_range = [(idx, acc) for idx, acc in enumerate(test_accs) if accuracy_min < acc < accuracy_max]

    # Define possible values for each position in the architecture (assumed binary or categorical)
    possible_values = [0, 1, 2, 3, 4]  # Adjust this based on your architecture's encoding scheme

    # Loop over each architecture in the desired accuracy range
    for idx, acc in architectures_in_range:
        #vector = architectures[idx]  # The architecture vector (e.g., a binary or categorical encoding)
        vector = bench.encode({'arch':bench.archive['str'][idx]})
        N = len(vector)  # Length of the architecture vector
        
        # Find neighbors with radius 3
        neighbors = neighbors_by_radius(N, possible_values, vector, radius)

        # Loop over neighbors and check if any have a test accuracy within the tolerance (1%)
        for neighbor in neighbors:
            arch = bench.decode(neighbor)
            neighbor_acc = bench.get_info_from_arch(arch)['test-acc']
            neighbor_idx = bench.archive['str'].index(arch['arch'])

            if abs(acc - neighbor_acc) <= tolerance:  # Check if the accuracy difference is within 1%
                print(f"Found neighbor for architecture {idx} (accuracy: {acc:.2f}):")
                print(f"Neighbor {neighbor_idx} with accuracy: {neighbor_acc:.2f}")
                return idx, neighbor_idx, acc, neighbor_acc

    print("No neighbor found with similar performance within the specified range and tolerance.")
    return None

def neighbors_by_radius(N, possible_values, vector, radius):
    neighbors = []
    for config in product(possible_values, repeat=N):
        diff_count = sum([1 for x, y in zip(vector, config) if x != y])
        if diff_count == radius:
            neighbors.append(config)
    return neighbors

def avg_test_acc(bench, configs):
    avg_test_acc = 0
    accs = []
    for c in configs:
        arch = bench.decode(c)
        test_acc = bench.get_info_from_arch(arch)['test-acc']
        accs.append(test_acc)
        avg_test_acc += test_acc
    return avg_test_acc/len(configs), accs, np.std(accs)

def get_targets_acc(bench):
    if bench.dataset == 'cifar10':
        max = 85 #avg acc
        min = 75 #bad acc
    elif bench.dataset == 'cifar100':
        max = 65
        min = 55
    else: # imagenet
        max = 35
        min = 25
    return max, min

def get_limits_plot(dataset):
    if dataset == 'cifar10':
        max = 95 #avg acc
        min = 70 #bad acc
    elif dataset == 'cifar100':
        max = 75
        min = 40
    else: # imagenet
        max = 50
        min = 10
    return max, min

def rank_by_test_acc(bench):
    '''
    if bench.dataset=='cifar10':
        val_dataset = bench.dataset + '-valid'
    else:
        val_dataset = bench.dataset
    '''
    # Return indexes of architectures with target test accuracies and the corresponding accuracies

    val_dataset = bench.dataset
    test_accs = bench.archive['test-acc'][val_dataset]
    idxs = list(range(len(test_accs)))
    test_accs_idxs = list(zip(test_accs, idxs))
    sorted_test_accs_idxs = sorted(test_accs_idxs, key=lambda x: x[0])
    sorted_test_accs = [test_acc for test_acc, _ in sorted_test_accs_idxs]
    sorted_idxs = [idx for _, idx in sorted_test_accs_idxs]
    # Filter the indices for architectures with specific validation accuracies
    filtered_idxs = []
    filtered_idxs.append(sorted_idxs[-1])  # Add the best architecture
    max,min=get_targets_acc(bench)
    found_max= False 
    found_min = False
    for test_acc, idx in reversed(sorted_test_accs_idxs):
        if not found_max and test_acc < max:
            filtered_idxs.append(idx)
            found_max = True
        
        if not found_min and test_acc < min:
            filtered_idxs.append(idx)
            found_min = True
        
        if found_max and found_min:
            break  # Stop if both conditions are met
    
    filtered_test_accs = [test_acc for test_acc, _ in sorted_test_accs_idxs if _ in filtered_idxs]

    return filtered_test_accs, filtered_idxs

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
    sorted_test_accs, sorted_idxs = rank_by_test_acc(bench)
    print("SORTED VAL ACCS: ", sorted_test_accs)

    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
    config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
    radius_range=range(1,4)
    # Placeholder lists to store accuracy values
    acc_config1 = [sorted_test_accs[2]]
    acc_config2 = [sorted_test_accs[1]]
    acc_config3 = [sorted_test_accs[0]]

    for radius in radius_range:
        # Calculate neighbors for each configuration
        neighbors_config1 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, radius)
        neighbors_config2 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config2, radius)
        neighbors_config3 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config3, radius)
        
        # Calculate average validation accuracy for each configuration
        avg_acc_config1 = avg_test_acc(bench, neighbors_config1)[0]
        avg_acc_config2 = avg_test_acc(bench, neighbors_config2)[0]
        avg_acc_config3 = avg_test_acc(bench, neighbors_config3)[0]
        
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
    sorted_test_accs, sorted_idxs = rank_by_test_acc(bench)
    print("SORTED VAL ACCS: ", sorted_test_accs)

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
        acc_neighbors_config1 = avg_test_acc(bench, neighbors_config1)[1]
        acc_neighbors_config2 = avg_test_acc(bench, neighbors_config2)[1]
        acc_neighbors_config3 = avg_test_acc(bench, neighbors_config3)[1]
        
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

def get_bins(dataset, radius):

    if dataset == 'cifar10':
        if radius == 1 :
            return 10, 100, 100
        elif radius == 2:
            return 10, 100, 100
        elif radius == 3:
            return 90, 90, 90
    elif dataset == 'cifar100':
        if radius == 1:
            return 10, 20, 30
        elif radius == 2:
            return 20, 40, 60
        elif radius == 3:
            return 60, 60, 60
    elif dataset == 'ImageNet16-120':
        if radius == 1 :
            return 10, 20, 30
        elif radius == 2:
            return 20, 40, 60
        elif radius == 3:
            return 60, 60, 60

def plot_histograms(data_array, bins=100, path='', baselines=None, dataset='cifar10', radius=1): #add radius

    FONT_SIZE = 8
    #FIGSIZE = (3.5, 3.0)
    FIGSIZE = (4,5)
    COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]
    titles = ['Model A', 'Model B', 'Model C']

    num_plots = len(data_array) - 1 

    # Set up subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=FIGSIZE, sharex=False)

    #print("DATA ARRAY: ", data_array)
    #print("ALL DATASET: ", data_array[0][:10])
    all_dataset = data_array[0]
    data_array = data_array[1:]
    # Plot histograms and curves for each element in the array
    for i, data in enumerate(data_array):
        temp_bins = get_bins(dataset, radius)[i]
        print("TEMP BINS: ", temp_bins)
        #print("DATA: ", data[:10])
        data = np.array(data)   
        data = data[data > 10]

        # Plot transparent curve behind histogram for the first element
        sns.histplot(all_dataset, bins=bins, color='green', edgecolor='black', kde=True, line_kws={'linewidth': 1, 'alpha': 0.2}, ax=axs[i], stat='density')
        sns.histplot(data, bins=temp_bins, color='darkblue', edgecolor='black', kde=True, line_kws={'linewidth': 2}, ax=axs[i], stat='density')
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)  # Hide y-axis values
        x_max,x_min = get_limits_plot(dataset)
        axs[i].set_xlim(x_min, x_max)

        # Set y-axis limits
        y_values = [bar.get_height() for bar in axs[i].patches]
        max_density = max(y_values) 
        max_density = max_density if max_density < 0.8 else 0.8 # Limit the maximum density to 0.8
        ymax = math.ceil(max_density * 10) / 10
        axs[i].set_ylim(0, ymax)  

        # Add title to each subplot
        axs[i].set_title(f"{titles[i]} (Test accuracy: {baselines[i]:.2f}%)")  # Adjust title as needed

        # Plot baselines as vertical lines
        if baselines:
                axs[i].axvline(baselines[i], color='red', linestyle='--', linewidth=1)
        
        # Add grid with custom interval
        axs[i].grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
        axs[i].set_yticks(np.arange(0, ymax, 0.2))

    # Add common X-axis label
    axs[-1].set_xlabel('Accuracy', fontsize=FONT_SIZE)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)

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
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)  
    sorted_test_accs, sorted_idxs = rank_by_test_acc(bench)
    test_accs = bench.archive['test-acc'][dataset]

    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
    config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
    print("CONFIG BAD ARCH: ", config3)

    configs = [config1, config2, config3]

    radius_range=range(1,4)
    #radius_range=range(1,2)
    # Placeholder lists to store accuracy values
    accuracies_by_radius_configs = []

    for idx in range(len(configs)):

        acc_neighbors_configs = []
        for radius in radius_range:
            model_path = os.path.join(result_dir,"test_accuracies_config_" + str(idx+1) + "_radius_" + str(radius) + ".npy")
            if not os.path.exists(model_path):
                print("Calculating array")
                config=configs[idx]
                # Calculate neighbors for each configuration
                neighbors_config = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config, radius)
                
                # Calculate validation accuracy for each configuration
                acc_neighbors_config = avg_test_acc(bench, neighbors_config)[1]
                
                acc_neighbors_config = np.array(acc_neighbors_config)

                # Save the numpy arrays to a single file
                np.save(model_path, acc_neighbors_config)

            else:
                print("Loading array")
                acc_neighbors_config = np.load(model_path)
                acc_neighbors_configs.append(acc_neighbors_config)
                #print("ACC NEIGHBORS CONFIG: ", acc_neighbors_config.shape)
        
        accuracies_by_radius_configs.append(acc_neighbors_configs)

    acc_by_configs = [test_accs, accuracies_by_radius_configs[0], accuracies_by_radius_configs[1], accuracies_by_radius_configs[2]]
    return acc_by_configs

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

def get_2good_archs(bench):

    # Returns best config and best config of neighbor of radius 3
    #net1 config
    sorted_test_accs, sorted_idxs = rank_by_test_acc(bench)
    if bench.dataset=='cifar10':
        config_idx = 0
    elif bench.dataset=='cifar100':
        config_idx = 1
    else:
        config_idx = 2
    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    arch = bench.decode(config1)
    acc1 = bench.get_info_from_arch(arch)
    #net2 config: give the config of neighbor of radius 3 of the first config with best test acc
    acc_radius3= compute_acc_by_radius(bench.dataset)[config_idx+1][2]
    new_data,new_indices = remove_outliers(acc_radius3)
    idx = new_indices[np.argmax(new_data)]
    config2 = list(neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, 3)[idx])
    arch=bench.decode(config2)
    acc2 = bench.get_info_from_arch(arch)
    return config1, config2, acc1, acc2

def get_good_bad_archs(bench):

    # Returns best config and worst config of neighbor of radius 3
    #net1 config
    sorted_test_accs, sorted_idxs = rank_by_test_acc(bench)
    if bench.dataset=='cifar10':
        config_idx = 0
    elif bench.dataset=='cifar100':
        config_idx = 1
    else:
        config_idx = 2
    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    arch = bench.decode(config1)
    acc1 = bench.get_info_from_arch(arch)
    #net2 config: give the config of neighbor of radius 3 of the first config with worst test acc
    acc_radius3= compute_acc_by_radius(bench.dataset)[config_idx+1][2]
    new_data,new_indices = remove_outliers(acc_radius3)
    idx = new_indices[np.argmin(new_data)]
    config2 = list(neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, 3)[idx])
    arch=bench.decode(config2)
    acc2 = bench.get_info_from_arch(arch)
    return config1, config2, acc1, acc2

def get_2bad_archs(bench):

    # Returns worst config and worst config of neighbor of radius 3
    #net1 config
    #sorted_test_accs, sorted_idxs = rank_by_test_acc(bench)

    #get worst test acc nasbench arch 
    '''
    idx = np.argmin(bench.archive['test-acc'][bench.dataset])
    if bench.dataset=='cifar10':
        config_idx = 0
    elif bench.dataset=='cifar100':
        config_idx = 1
    else:
        config_idx = 2
    config1 = bench.encode({'arch':bench.archive['str'][idx]})
    '''
    #config1 = get_bad_arch(bench.dataset)
    #config1 = bench.encode({'arch':bench.archive['str'][idx]})
    #arch = bench.decode(config1)
    #acc1 = bench.get_info_from_arch(arch)
    #net2 config: give the config in the neighborhood of radius 3 of the third config (bad arch) with worst test acc
    _, sorted_idxs = rank_by_test_acc(bench)
    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
    arch=bench.decode(config1)
    acc1 = bench.get_info_from_arch(arch)
    _,_,_,acc_neighbor_config3= compute_acc_by_radius(bench.dataset)
    acc_radius3 = acc_neighbor_config3[2] #accs of neighbor radius 3 of config1
    print("ACC1: ", acc1)
    new_data,new_indices = remove_outliers(acc_radius3)
    idx = new_indices[np.argmin(new_data)]
    config2 = list(neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, 3)[idx])
    arch=bench.decode(config2)
    acc2 = bench.get_info_from_arch(arch)
    print("ACC2: ", acc2)
    return config1, config2, acc1, acc2

def path_bench(dataset,quality):
    print("DATASET: ", dataset)
    bench = NASBench201(dataset=dataset)
    config1, config2, acc1, acc2 = get_archs(dataset, quality)
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
    avg_test_accs = []
    std_test_accs = []
    for configs in archs_by_level:
        avg, _, std = avg_test_acc(bench, configs)
        avg_test_accs.append(avg)
        std_test_accs.append(std)
    path_accs= [acc1] + avg_test_accs + [acc2]
    print("PATH ACCS: ", path_accs)
    print("STD VAL ACCS: ", std_test_accs)
    # Plot the path
    # (flat plot is better)
    # Plot the path
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

    if not os.path.exists('../results/flatness_exp'):
        os.makedirs('results/flatness_exp', exist_ok=True)
    plt.savefig('results/flatness_exp/'+dataset+'/path_accs_' + quality +'.pdf', format='pdf', bbox_inches='tight', dpi=300)

     
def compute_barrier(accs):
    acc_a = accs[0]
    acc_b = accs[1]
    acc_min = min(accs)
    return np.round(0.5 * (acc_a + acc_b) - acc_min,2)

def path_bench_qualities(dataset):
    import os
    import json
    import matplotlib.pyplot as plt

    print("DATASET: ", dataset)
    FONT_SIZE = 18
    bench = NASBench201(dataset=dataset)

    # Define qualities and their respective colors
    qualities = ["Model A", "Model B", "Model C"]
    colors = ["purple", "orange", "green"]
    paths_by_quality = {}
    accs_by_quality = {}
    std_by_quality = {}
    barriers = []

    for quality in qualities:
        config1, config2, acc1, acc2 = get_archs(dataset, quality)
        # Find the paths between the two configurations
        paths = search_tree(config1, config2)
        max_level = len(paths[0]) - 2
        archs_by_level = [[] for _ in range(max_level)]
        
        # Collect architectures by level
        for i in range(max_level):
            for path in paths:
                archs_by_level[i].append(path[i + 1])

        # Average validation accuracy by level
        avg_test_accs = []
        std_test_accs = []
        for configs in archs_by_level:
            avg, _, std = avg_test_acc(bench, configs)
            avg_test_accs.append(avg)
            std_test_accs.append(std)
        
        # Store path accuracies and standard deviations for this quality
        path_accs = [acc1] + avg_test_accs + [acc2]
        paths_by_quality[quality] = path_accs
        std_by_quality[quality] = [0] + std_test_accs + [0]  # no std dev for acc1 and acc2
        accs_by_quality[quality] = (acc1, acc2)
        barriers.append(compute_barrier(path_accs))

        print(f"PATH ACCS for {quality}: ", path_accs)
        print(f"STD VAL ACCS for {quality}: ", std_test_accs)

    # Plot the paths for Model A, Model B, and Model C qualities
    plt.figure(figsize=(10, 5))
    x = list(range(4))  # 0, 1, 2, 3 representing the radius

    for quality, color in zip(qualities, colors):
        y = paths_by_quality[quality]
        yerr = std_by_quality[quality]
        plt.errorbar(
            x, y, yerr=yerr, fmt='-o', capsize=5, capthick=2, elinewidth=1, color=color, label=f'{quality}'
        )
        # Add squares for extreme points
        plt.scatter([x[0], x[-1]], [y[0], y[-1]], color=color, s=80, marker='s', zorder=3)  # Squares
        # Add circles for internal points
        plt.scatter(x[1:-1], y[1:-1], color=color, s=80, marker='o', zorder=3)  # Circles

    # Customize plot
    plt.xlabel('Radius', fontsize=FONT_SIZE + 2)
    plt.ylabel('Accuracy', fontsize=FONT_SIZE + 2)
    if dataset == 'cifar10': 
        name = 'CIFAR-10'
    elif dataset == 'cifar100':
        name = 'CIFAR-100'
    elif dataset == 'ImageNet16-120':
        name = 'ImageNet16-120'
    plt.title(f'Path Accuracies for {name}', fontsize=FONT_SIZE + 4)
    plt.xticks(x, fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.show()

    # Save the plot to the results folder
    results_dir = f'results/flatness_exp_{dataset}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    plot_path = f'{results_dir}/path_accs_all_qualities_'+dataset+'.pdf'
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)

    # Save barriers in a JSON file
    barriers_path = f'{results_dir}/barriers.json'
    with open(barriers_path, 'w') as f:
        json.dump(barriers, f)


def plot_histo_configs_radius1(dataset, radius=1):
    bench=NASBench201(dataset=dataset)
    sorted_test_accs, sorted_idxs = rank_by_test_acc(bench)
    accs = compute_acc_by_radius(dataset=dataset)
    result_dir = f'results/flatness_exp_{dataset}'
    # Save the plot to the results folder
    if not os.path.exists(f'results/flatness_exp_{dataset}'):
        os.makedirs(f'results/flatness_exp_{dataset}', exist_ok=True)
    # test_accs
    plot_histograms([accs[0],accs[1][radius-1], accs[2][radius-1], accs[3][radius-1]], path=os.path.join(result_dir,'histogram_config'+'_'+dataset+'_radius' + str(radius)+'.pdf'), baselines=(sorted_test_accs)[::-1], dataset=dataset, radius=radius) 

def get_archs(dataset, quality):
    bench=NASBench201(dataset=dataset)
    idx1,idx2=0,0
    if dataset=='cifar10':
        if quality=='Model A':
            idx1=81
            idx2=1459
        elif quality=='Model B':
            idx1=0
            idx2=163
        elif quality=='Model C':
            idx1=40
            idx2=12094
    elif dataset=='cifar100':
        if quality=='Model A':
            idx1=81
            idx2=11711
        elif quality=='Model B':
            idx1=31
            idx2=7680
        elif quality=='Model C':
            idx1=146
            idx2=4461
    elif dataset=='ImageNet16-120':
        if quality=='Model A':
            idx1=65
            idx2=2246
        elif quality=='Model B':
            idx1=17
            idx2=5845
        elif quality=='Model C':
            idx1=2
            idx2=348
    return bench.encode({'arch':bench.archive['str'][idx1]}), bench.encode({'arch':bench.archive['str'][idx2]}), bench.get_info_from_arch({'arch':bench.archive['str'][idx1]})['test-acc'], bench.get_info_from_arch({'arch':bench.archive['str'][idx2]})['test-acc']


def get_idx_interval(acc, dataset):

    if dataset == 'cifar10':
        if acc < 94.44 and acc > 94.3:
            return 0
        elif acc < 85.7 and acc > 84.3:
            return 1
        elif acc < 75.7 and acc > 74.3:
            return 2
        else:
            return -1
    elif dataset == 'cifar100':
        if acc < 74.2 and acc > 68.8 :
            return 0
        elif acc < 65.7 and acc > 64.3:
            return 1
        elif acc < 55.7 and acc > 54.3:
            return 2
        else:
            return -1
    elif dataset == 'ImageNet16-120':
        if acc < 48 and acc > 46.6:
            return 0
        elif acc < 35.7 and acc > 34.3:
            return 1
        elif acc < 25.7 and acc > 24.3:
            return 2
        else:
            return -1

def get_baselines(dataset):
    if dataset == 'cifar10':
        return [94.37, 85, 75]
    elif dataset == 'cifar100':
        return [73.5, 65, 55]
    elif dataset == 'ImageNet16-120':
        return [47.3, 35, 25]

import os
import numpy as np

def distributions_nasbench(bench, dataset, radius, dist_path='results/flatness_exp'):
    # Retrieve test accuracies for the dataset
    test_accs = bench.archive['test-acc'][dataset]
    print("LEN TEST ACCS: ", len(test_accs))
    dist = [[] for _ in range(3)]  
    dist.insert(0, test_accs)
    folder = os.path.join(dist_path, 'nasbenchdist_' + dataset + '_radius' + str(radius))

    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)

    # Check if precomputed distributions exist
    if os.path.exists(os.path.join(folder,'neighborsnet_0.npy')):
        for i in range(3):
            dist[i] = np.load(os.path.join(folder,'neighborsnet_' + str(i) + '.npy'), allow_pickle=True)
    else:
        # Compute distributions
        for id_net in range(len(test_accs)):

            acc=test_accs[id_net]
            dist_idx = get_idx_interval(acc, dataset)
            #print("DIST IDX: ", dist_idx)
            if dist_idx == -1:
                continue  # Skip if outside defined intervals

            # Convert architecture to vector and compute neighbors
            config = bench.archive['str'][id_net]
            config_vector = bench.encode({'arch': config})
            neighbors_config = neighbors_by_radius(
                bench.nvar, list(range(bench.num_operations)), config_vector, radius
            )

            # Compute average accuracies of neighbors
            acc_neighbors_config = avg_test_acc(bench, neighbors_config)[1]
            dist[dist_idx].extend(acc_neighbors_config)

        # Save computed distributions
        for i in range(3):
            np.save(os.path.join(folder, 'neighborsnet_' + str(i) + '.npy'), dist[i])

    # Plot histograms
    plot_histograms(
        dist, bins=100,
        path=os.path.join(folder, f'histo_nasbench_{dataset}_{radius}.pdf'),
        baselines=get_baselines(dataset),
        dataset=dataset,
        radius=radius
    )


bench = NASBench201(dataset='cifar10')
distributions_nasbench(bench, 'cifar10', 1)
distributions_nasbench(bench, 'cifar10', 2)
distributions_nasbench(bench, 'cifar10', 3)
bench = NASBench201(dataset='cifar100')
distributions_nasbench(bench, 'cifar100', 1)
distributions_nasbench(bench, 'cifar100', 2)
distributions_nasbench(bench, 'cifar100', 3)
bench = NASBench201(dataset='ImageNet16-120')
distributions_nasbench(bench, 'ImageNet16-120', 1)
distributions_nasbench(bench, 'ImageNet16-120', 2)
distributions_nasbench(bench, 'ImageNet16-120', 3)

'''

dataset='ImageNet16-120' #
bench=NASBench201(dataset=dataset)
idx1, idx2, acc1, acc2 = find_neighbors_with_similar_performance(bench, 27, 28.5, radius=3, tolerance=0.1)
print(idx1, idx2)
print(acc1, acc2)
print(bench.archive['str'][idx1])
print(bench.archive['str'][idx2])
'''

'''
path_bench_qualities('cifar10')
path_bench_qualities('cifar100')
path_bench_qualities('ImageNet16-120')
'''


'''
path_bench('cifar100')
path_bench('ImageNet16-120')
'''

#plot_histo_configs_radius1('cifar10',1)
#plot_histo_configs_radius1('cifar100',1)
#plot_histo_configs_radius1('ImageNet16-120',1) 

#compute_acc_by_radius('cifar10')
#compute_acc_by_radius('cifar100')
#compute_acc_by_radius('ImageNet16-120')


