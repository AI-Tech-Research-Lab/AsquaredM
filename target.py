import itertools
import random

class GenotypeModifier:
    def __init__(self, operations):
        self.operations = operations  # List of available operations, e.g., ['sep_conv_3x3', 'skip_connect', ...]

    def apply_targeted_actions(self, current_genotype, target_genotype, num_actions):
        """
        Applies actions to modify the current genotype to move closer to the target genotype.
        Each action changes an operation or connection based on the target genotype.
        """
        modified_genotype = {
            'normal': current_genotype['normal'][:],
            'normal_concat': current_genotype['normal_concat'][:],
            'reduce': current_genotype['reduce'][:],
            'reduce_concat': current_genotype['reduce_concat'][:]
        }

        # Collect differences between current and target genotype
        differences = []
        for section in ['normal', 'reduce']:
            for i, (current_op, current_node) in enumerate(modified_genotype[section]):
                target_op, target_node = target_genotype[section][i]
                if current_op != target_op or current_node != target_node:
                    differences.append((section, i, current_op, current_node, target_op, target_node))

        # Apply actions up to the specified number
        actions_applied = 0
        while actions_applied < num_actions and differences:
            # Select a random difference to apply
            section, i, current_op, current_node, target_op, target_node = random.choice(differences)

            # Apply the change to match the target configuration
            if current_op != target_op:
                # Change the operation
                modified_genotype[section][i] = (target_op, current_node)
            elif current_node != target_node:
                # Change the node connection
                modified_genotype[section][i] = (current_op, target_node)

            # Remove the applied change from differences
            differences = [(sec, idx, cur_op, cur_node, tar_op, tar_node)
                           for sec, idx, cur_op, cur_node, tar_op, tar_node in differences
                           if not (sec == section and idx == i)]

            actions_applied += 1

        return modified_genotype
    
    def apply_targeted_actionsV2(self, current_genotype, target_genotype, num_actions):
        """
        Returns all possible genotypes obtained from the current genotype by applying
        up to num_actions changes to move closer to the target genotype.
        """
        modified_genotypes = []

        # Collect differences between current and target genotype
        differences = []
        for section in ['normal', 'reduce']:
            for i, (current_op, current_node) in enumerate(current_genotype[section]):
                target_op, target_node = target_genotype[section][i]
                if current_op != target_op or current_node != target_node:
                    differences.append((section, i, current_op, current_node, target_op, target_node))

        # Generate all combinations of changes up to num_actions
        for r in range(1, num_actions + 1):
            for changes in itertools.combinations(differences, r):
                print("r:", r)
                # Create a modified version of the current genotype
                modified_genotype = {
                    'normal': current_genotype['normal'][:],
                    'normal_concat': current_genotype['normal_concat'][:],
                    'reduce': current_genotype['reduce'][:],
                    'reduce_concat': current_genotype['reduce_concat'][:]
                }

                for section, i, current_op, current_node, target_op, target_node in changes:
                    print("changes:", changes)
                    if current_op != target_op:
                        # Change the operation
                        modified_genotype[section][i] = (target_op, current_node)
                    elif current_node != target_node:
                        # Change the node connection
                        modified_genotype[section][i] = (current_op, target_node)

                modified_genotypes.append(modified_genotype)

        return modified_genotypes

def count_differences(genotype1, genotype2):
    """Counts the differences between two genotype representations."""
    differences = []
    for section in ['normal', 'reduce']:
        for i, (op1, node1) in enumerate(genotype1[section]):
            op2, node2 = genotype2[section][i]
            if (op1, node1) != (op2, node2):
                differences.append((section, i, (op1, node1), (op2, node2)))
    return differences

# Usage example:
# List of available operations for demonstration purposes
available_operations = ['sep_conv_3x3', 'skip_connect', 'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3']

# Current genotype representation
current_genotype = {
    'normal': [('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 1),
               ('skip_connect', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 3), ('skip_connect', 0)],
    'normal_concat': list(range(2, 6)),
    'reduce': [('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0),
               ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)],
    'reduce_concat': list(range(2, 6))
}

# Target genotype representation
target_genotype = {
    'normal': [('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_5x5', 1),
               ('skip_connect', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 3), ('skip_connect', 1)],
    'normal_concat': [2, 3, 4, 5],
    'reduce': [('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0),
               ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)],
    'reduce_concat': [2, 3, 4, 5]
}

# Compare the original and modified genotypes
differences = count_differences(current_genotype, target_genotype)

# Print the results
print("Number of differences:", len(differences))
print("Differences:", differences)

# Initialize the modifier
modifier = GenotypeModifier(operations=available_operations)

# Apply a specified number of actions (e.g., 3) to move towards the target
new_genotypes = modifier.apply_targeted_actionsV2(current_genotype, target_genotype, num_actions=2)

print("Modified Genotype:")
print(new_genotypes)
print(len(new_genotypes))

'''
# Compare the original and modified genotypes
differences = count_differences(current_genotype, new_genotype)

# Print the results
print("Number of differences:", len(differences))
print("Differences:", differences)

# Compare the original and modified genotypes
differences = count_differences(target_genotype, new_genotype)

# Print the results
print("Number of differences:", len(differences))
print("Differences:", differences)
'''