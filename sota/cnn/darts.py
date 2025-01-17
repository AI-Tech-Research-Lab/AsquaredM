from itertools import combinations
import itertools
import os
import random
import sys
import numpy as np
sys.path.append('/u01/homes/fpittorino/workspace/darts-SAM')
from sota.cnn.spaces import PRIMITIVES

class DARTS():

    def __init__(self, topk):
        self.input_nodes = 2
        self.output_node = 1
        self.topk = topk
        self.operations = len(PRIMITIVES)
        self.int_nodes = 4
        n_archs = 0
        for i in range(0, self.int_nodes):
            n_archs += (self.input_nodes + i)
        self.n_archs = n_archs

    def sample(self, n_samples):
        archs = []
        matrix_size = self.input_nodes + self.int_nodes - 1
        for _ in range(n_samples):
            # Create random adjacency matrices for normal and reduce cells
            normal_adjacency_matrix = self._sample_single_matrix(matrix_size)
            reduce_adjacency_matrix = self._sample_single_matrix(matrix_size)
            archs.append((normal_adjacency_matrix, reduce_adjacency_matrix))
        return archs

    def _sample_single_matrix(self, matrix_size):
        adjacency_matrix = np.zeros((self.int_nodes, matrix_size), dtype=int)

        # Link first intermediate node to the first two input nodes
        for i in range(self.input_nodes):
            adjacency_matrix[0][i] = np.random.randint(0, self.operations)

        # Fill in edges for subsequent nodes
        for i in range(1, self.int_nodes):
            # Determine the range of preceding nodes for each row
            preceding_nodes = min(self.topk, self.input_nodes + i)  # Limit to topk or number of previous nodes + input nodes
            # Select preceding nodes for incoming edges
            incoming_edges = np.random.choice(self.input_nodes + i, preceding_nodes, replace=False)
            for j in incoming_edges:
                # Adjust the index for the connection to preceding nodes
                adjacency_matrix[i][j] = np.random.randint(1, self.operations)
        return adjacency_matrix

    def _count_differences(self, matrix1, original_matrix):
        # Crea una maschera per le celle non zero in original_matrix
        mask = original_matrix != 0
        
        # Applica la maschera alle differenze tra matrix1 e matrix2
        differences = (matrix1 != original_matrix) & mask
        
        # Conta le differenze
        return np.sum(differences)

    def create_neighboring_matrix(self, adjacency_matrices, radius):
        normal_matrix, reduce_matrix = adjacency_matrices

        # Randomly distribute the total number of changes between normal and reduce matrices
        normal_changes = np.random.randint(0, radius + 1)
        reduce_changes = radius - normal_changes

        initial_normal_matrix = normal_matrix.copy()
        initial_reduce_matrix = reduce_matrix.copy()

        # Applicazione delle modifiche alla matrice normal_matrix
        current_differences_normal = 0
        while current_differences_normal < normal_changes:
            normal_matrix = self._create_single_neighboring_matrix(normal_matrix)
            current_differences_normal = self._count_differences(initial_normal_matrix, normal_matrix)
        
        # Applicazione delle modifiche alla matrice reduce_matrix
        current_differences_reduce = 0
        while current_differences_reduce < reduce_changes:
            reduce_matrix = self._create_single_neighboring_matrix(reduce_matrix)
            current_differences_reduce = self._count_differences(initial_reduce_matrix, reduce_matrix)

        return (normal_matrix, reduce_matrix)

    def _create_single_neighboring_matrix(self, adjacency_matrix):
        int_nodes, matrix_size = adjacency_matrix.shape
        neighboring_matrix = np.copy(adjacency_matrix)
        # choose a random int node
        i = np.random.randint(0, int_nodes)
        # Choose a random edge from the selected node to alter
        edge_to_alter = np.random.randint(0, self.input_nodes + i)
        # If the edge is already present, change its operation
        if neighboring_matrix[i, edge_to_alter] != 0:
            current_operation = neighboring_matrix[i, edge_to_alter]
            new_operation = np.random.randint(1, self.operations)
            
            # Ensure the new operation is different from the current one
            while new_operation == current_operation:
                new_operation = np.random.randint(1, self.operations)
            
            neighboring_matrix[i, edge_to_alter] = new_operation
        else:
            # Find all existing edges in the same row
            non_zero_edges = np.nonzero(neighboring_matrix[i])[0]
            # Remove a random existing edge
            edge_to_zeroize = np.random.choice(non_zero_edges)
            neighboring_matrix[i, edge_to_zeroize] = 0  # Zeroize the edge
            # Choose a random operation for the new edge
            new_operation = np.random.randint(1, self.operations)
            neighboring_matrix[i, edge_to_alter] = new_operation

        return neighboring_matrix

    def encode_adjacency_matrix(self, adjacency_matrices):
        normal_matrix, reduce_matrix = adjacency_matrices
        normal_encoding = self._encode_single_matrix(normal_matrix)
        reduce_encoding = self._encode_single_matrix(reduce_matrix)
        return normal_encoding + reduce_encoding

    def _encode_single_matrix(self, adjacency_matrix):
        num_intermediate_nodes = adjacency_matrix.shape[0]
        op_input1 = adjacency_matrix[0][0]
        op_input2 = adjacency_matrix[0][1]
        ops_vector = [op_input1, op_input2]
        dag_vector = []
        num_input_nodes = self.input_nodes
        for i in range(1, num_intermediate_nodes):
            # Intermediate nodes
            preceding_nodes = np.nonzero(adjacency_matrix[i])[0]
            # Sort preceding nodes
            preceding_nodes.sort()
            # Create the encoding
            for node in range(i + num_input_nodes):
                if node in preceding_nodes:
                    dag_vector.extend([1])
                else:
                    dag_vector.extend([0])
            for node in preceding_nodes:
                ops_vector.extend([adjacency_matrix[i][node]])  # Add a non-zero value and operation
        return dag_vector + ops_vector

    def decode_to_adjacency_matrix(self, encoded_vector):
        # Split the vector into normal and reduce parts
        mid_point = len(encoded_vector) // 2
        normal_vector = encoded_vector[:mid_point]
        reduce_vector = encoded_vector[mid_point:]

        normal_matrix = self._decode_single_matrix(normal_vector)
        reduce_matrix = self._decode_single_matrix(reduce_vector)
        return (normal_matrix, reduce_matrix)

    def _decode_single_matrix(self, encoded_vector):
        num_input_nodes = self.input_nodes
        num_intermediate_nodes = self.int_nodes
        num_ops = 2 * num_intermediate_nodes
        offset = len(encoded_vector) - num_ops + 2

        # Initialize the adjacency matrix
        adjacency_matrix = np.zeros((num_intermediate_nodes, num_input_nodes + num_intermediate_nodes - 1), dtype=int)

        # Fill in the adjacency matrix using the encoded vector
        idx = 0
        j = 0
        for i in range(num_intermediate_nodes):
            # Add connections to input nodes for the first intermediate node
            if i == 0:
                adjacency_matrix[i][:num_input_nodes] = encoded_vector[-num_ops:-num_ops + 2]
            else:
                # Find the positions of 1s in the encoding indicating connections
                preceding_nodes = i + num_input_nodes
                connection_positions = [j for j, val in enumerate(encoded_vector[idx:idx + preceding_nodes]) if val == 1]
                # Fill in the adjacency matrix based on the connections
                for pos in connection_positions:
                    adjacency_matrix[i][pos] = encoded_vector[offset + j]
                    j += 1
                idx += preceding_nodes

        return adjacency_matrix

    def genotype_to_adjacency_matrix(self, genotype):
        #print("Genotype: ", genotype)
        normal_matrix = self._genotype_to_single_matrix(genotype['normal'])
        reduce_matrix = self._genotype_to_single_matrix(genotype['reduce'])
        return (normal_matrix, reduce_matrix)

    def _genotype_to_single_matrix(self, gene):
        adjacency_matrix = np.zeros((self.int_nodes, self.input_nodes + self.int_nodes - 1), dtype=int)
        for idx, (op, node) in enumerate(gene):
            node_pos = idx // 2
            op_idx = PRIMITIVES.index(op)
            adjacency_matrix[node_pos][node] = op_idx + 1
        return adjacency_matrix

    def adjacency_matrix_to_genotype(self, adjacency_matrices):
        #print("Adjacency Matrices: ", adjacency_matrices)
        normal_matrix, reduce_matrix = adjacency_matrices
        normal_gene = self._matrix_to_genotype(normal_matrix)
        reduce_gene = self._matrix_to_genotype(reduce_matrix)
        return {'normal': normal_gene, 'normal_concat': range(2, 6), 'reduce': reduce_gene, 'reduce_concat': range(2, 6)}

    def _matrix_to_genotype(self, adjacency_matrix):
        gene = []
        for i in range(self.int_nodes):
            for j in range(self.input_nodes + i):
                op = adjacency_matrix[i][j]
                if op > 0:
                    gene.append((PRIMITIVES[op-1], j))
        return gene

    def genotype_to_vector(self, genotype):
        adjacency_matrices = self.genotype_to_adjacency_matrix(genotype)
        encoded_vector = self.encode_adjacency_matrix(adjacency_matrices)
        return encoded_vector

    def vector_to_genotype(self, encoded_vector):
        adjacency_matrices = self.decode_to_adjacency_matrix(encoded_vector)
        genotype = self.adjacency_matrix_to_genotype(adjacency_matrices)
        return genotype
    
    '''
    def sample_neighbors(self, adjacency_matrices, radius, n):
        unique_matrices = set()
        normal_matrix, reduce_matrix = adjacency_matrices

        while len(unique_matrices) < n:
            neighbor_pair = self.create_neighboring_matrix((normal_matrix, reduce_matrix), radius)
            # Convert matrices to tuples to be hashable
            neighbor_tuple = (tuple(neighbor_pair[0].flatten()), tuple(neighbor_pair[1].flatten()))
            unique_matrices.add(neighbor_tuple)

        # Convert tuples back to matrices
        genes = []
        #red_matrices = []
        for mat_tuple in unique_matrices:
            norm_matrix = np.array(mat_tuple[0]).reshape(self.int_nodes, self.input_nodes + self.int_nodes - 1)
            red_matrix = np.array(mat_tuple[1]).reshape(self.int_nodes, self.input_nodes + self.int_nodes - 1)
            genes.append(self.adjacency_matrix_to_genotype((norm_matrix,red_matrix)))

        return genes
    '''
    def sample_neighbors(self, adjacency_matrices, radius, n):
        unique_neighbors = set()
        normal_matrix, reduce_matrix = adjacency_matrices

        while len(unique_neighbors) < n:
            # Generate a neighboring matrix pair
            neighbor_matrices = self.create_neighboring_matrix((normal_matrix, reduce_matrix), radius)
            normal_neighbor, reduce_neighbor = neighbor_matrices

            # Add checks for each row in the matrices
            if not self._validate_matrix_rows(normal_neighbor) or not self._validate_matrix_rows(reduce_neighbor):
                print("Invalid matrix")
                continue  # Skip invalid matrices

            # Create a hashable representation of the matrices
            neighbor_tuple = (
                tuple(normal_neighbor.flatten()),
                tuple(reduce_neighbor.flatten())
            )

            # Add to the set if unique
            if neighbor_tuple not in unique_neighbors:
                unique_neighbors.add(neighbor_tuple)

        # Convert tuples back to adjacency matrices and return genotypes
        genotypes = []
        for neighbor_tuple in unique_neighbors:
            norm_matrix = np.array(neighbor_tuple[0]).reshape(self.int_nodes, self.input_nodes + self.int_nodes - 1)
            red_matrix = np.array(neighbor_tuple[1]).reshape(self.int_nodes, self.input_nodes + self.int_nodes - 1)
            genotypes.append(self.adjacency_matrix_to_genotype((norm_matrix, red_matrix)))

        return genotypes

    def _validate_matrix_rows(self, adjacency_matrix):
        """
        Validate each row of the adjacency matrix to ensure it satisfies constraints:
        - At least two non-zero operations per row.
        - Optional: Validate operation indices are within a valid range.
        """
        for row in adjacency_matrix:
            # Check if the row has at least two non-zero operations
            if np.count_nonzero(row) < 2:
                return False

            # Check for valid operation indices (optional)
            if not all(0 <= op < self.operations for op in row if op != 0):
                return False

        return True

    
    def to_dict(self, genotype):
        return genotype._asdict()

    def sample_neighbors_path(self, current_genotype, target_genotype, num_actions):
        import copy
        modified_genotypes = []

        # Reorder target genotype for alignment
        for section in ['normal', 'reduce']:
            for i in range(0, len(target_genotype[section]) - 1, 2):  # Step by 2
                first_op, first_node = target_genotype[section][i]
                second_op, second_node = target_genotype[section][i + 1]
                if (first_op, first_node) == current_genotype[section][i + 1]:
                    target_genotype[section][i], target_genotype[section][i + 1] = \
                        target_genotype[section][i + 1], target_genotype[section][i]
                elif (second_op, second_node) == current_genotype[section][i]:
                    target_genotype[section][i], target_genotype[section][i + 1] = \
                        target_genotype[section][i + 1], target_genotype[section][i]

        # Collect differences
        differences = []
        for section in ['normal', 'reduce']:
            for i, (current_op, current_node) in enumerate(current_genotype[section]):
                target_op, target_node = target_genotype[section][i]
                if current_op != target_op or current_node != target_node:
                    differences.append((section, i, current_op, current_node, target_op, target_node))

        # Generate all combinations of changes
        for r in range(1, num_actions + 1):
            for changes in itertools.combinations(differences, r):
                modified_genotype = copy.deepcopy(current_genotype)

                for section, i, current_op, current_node, target_op, target_node in changes:
                    if current_op != target_op:
                        modified_genotype[section][i] = (target_op, current_node)
                    elif current_node != target_node:
                        modified_genotype[section][i] = (current_op, target_node)

                # Validate and append
                if len(modified_genotype['normal']) == 8 and len(modified_genotype['reduce']) == 8:
                    modified_genotypes.append(modified_genotype)

        return modified_genotypes

    
    '''
    def sample_neighbors_path(self, current_genotype, target_genotype, num_actions):
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
    '''
    
'''
    def sample_neighbors_path(self,init_config, target_config):
        """
        Generate a list of lists where each sublist contains all possible configurations with exactly K total differences,
        for K ranging from 1 to max_possible_differences.

        Args:
        - init_config (tuple): A tuple containing two matrices (normal_matrix, reduce_matrix) representing the initial configuration.
        - target_config (tuple): A tuple containing two matrices (normal_matrix, reduce_matrix) representing the target configuration.

        Returns:
        - configs_list (list of lists): A list of lists, where each sublist contains tuples of matrices representing the configurations.
        
        Number of combinations for each k is equal to the binomial coefficent (n,k) where n is the total number of differences
        """
        init_normal_matrix, init_reduce_matrix = init_config
        target_normal_matrix, target_reduce_matrix = target_config

        def get_differences(init_matrix, target_matrix):
            """
            Get the indices where init_matrix and target_matrix differ.
            """
            return np.argwhere(init_matrix != target_matrix)

        def apply_changes(init_matrix, target_matrix, indices):
            """
            Apply changes to the init_matrix based on the indices to be changed.
            """
            new_matrix = init_matrix.copy()
            for row, col in indices:
                new_matrix[row, col] = target_matrix[row, col]
            return new_matrix

        def generate_configurations_for_k(init_matrix, target_matrix, k):
            """
            Generate all possible matrices with exactly k differences.
            """
            differences = get_differences(init_matrix, target_matrix)
            configurations = []
            for indices in combinations(differences, k):
                modified_matrix = apply_changes(init_matrix, target_matrix, indices)
                configurations.append(modified_matrix)
            return configurations

        # Get the differences for both matrices
        normal_differences = get_differences(init_normal_matrix, target_normal_matrix)
        reduce_differences = get_differences(init_reduce_matrix, target_reduce_matrix)
        max_normal_changes = len(normal_differences)
        max_reduce_changes = len(reduce_differences)

        print("MAX NORMAL CHANGES: ", max_normal_changes)
        print("MAX REDUCE CHANGES: ", max_reduce_changes)

        # Compute all possible matrices for different values of K
        all_configs = []
        for k in range(1, max_normal_changes + max_reduce_changes):
            configs_for_k = []
            # Generate configurations for each distribution of K changes between normal and reduce matrices
            for normal_changes in range(max(0, k - max_reduce_changes), min(k, max_normal_changes) + 1):
                reduce_changes = k - normal_changes
                if reduce_changes > max_reduce_changes:
                    continue
                
                normal_configs_for_k = generate_configurations_for_k(init_normal_matrix, target_normal_matrix, normal_changes)
                reduce_configs_for_k = generate_configurations_for_k(init_reduce_matrix, target_reduce_matrix, reduce_changes)
                
                for normal_matrix in normal_configs_for_k:
                    for reduce_matrix in reduce_configs_for_k:
                        #configs_for_k.append((normal_matrix, reduce_matrix))
                        configs_for_k.append(self.adjacency_matrix_to_genotype((normal_matrix,reduce_matrix)))
            
            all_configs.append(configs_for_k)

        return all_configs, max_normal_changes + max_reduce_changes

'''
'''
darts = DARTS(2)
#matrix = darts.sample(1)[0]
#print(matrix[0])
#print(matrix[1])


normal_matrix = np.array([
    [3, 3, 0, 0, 0],
    [0, 5, 2, 0, 0],
    [0, 0, 3, 1, 0],
    [0, 0, 1, 0, 6]
])

reduce_matrix = np.array([
    [2, 6, 0, 0, 0],
    [2, 0, 3, 0, 0],
    [5, 0, 5, 0, 0],
    [0, 6, 0, 0, 4]
])

print("Normal Matrix:")
print(normal_matrix)
print("\nReduce Matrix:")
print(reduce_matrix)


matrix = (normal_matrix, reduce_matrix)

normal_matrix = np.array([
    [3, 4, 0, 0, 0], # difference in 2nd entry
    [0, 5, 2, 0, 0],
    [0, 0, 3, 1, 0],
    [0, 0, 3, 0, 5] #difference in3rd entry
])

reduce_matrix = np.array([
    [2, 6, 0, 0, 0],
    [2, 0, 3, 0, 0],
    [5, 0, 6, 0, 0], # difference in 3rd entry
    [0, 2, 0, 0, 4]  # difference in 2nd entry
])

configs = darts.sample_neighbors_path(matrix, (normal_matrix, reduce_matrix))

print("K=1")
print(len(configs[0]))
print(configs[0])

print("K=2")
print(len(configs[1]))
print(configs[1])

print("K=3")
print(len(configs[2]))
print(configs[2])
'''

'''
gene = darts.adjacency_matrix_to_genotype(matrix)
print("Gene:")
print(gene)
matrix= darts.genotype_to_adjacency_matrix(gene)
print("Matrix:")
print(matrix)
'''

'''
neighbors = darts.sample_neighbors(matrix, 1, 5)
print("Neighbors:")
for neighbor in neighbors:
    print(neighbor)


vector = darts.encode_adjacency_matrix(matrix)
print(vector)
#matrix = darts.decode_to_adjacency_matrix(vector)
#print(matrix)
matrix2 = darts.create_neighboring_matrix(matrix, 3)
print(matrix2[0])
print(matrix2[1])
vector2 = darts.encode_adjacency_matrix(matrix2)
print(vector2)
'''
