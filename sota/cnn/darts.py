import os
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

    def create_neighboring_matrix(self, adjacency_matrices, radius):
        normal_matrix, reduce_matrix = adjacency_matrices

        # Randomly distribute the total number of changes between normal and reduce matrices
        normal_changes = np.random.randint(0, radius + 1)
        reduce_changes = radius - normal_changes
        #print("Normal Changes: ", normal_changes)

        for _ in range(normal_changes):
            normal_matrix = self._create_single_neighboring_matrix(normal_matrix)
        for _ in range(reduce_changes):
            reduce_matrix = self._create_single_neighboring_matrix(reduce_matrix)

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
    
    def to_dict(self, genotype):
        return genotype._asdict()

'''
darts = DARTS(2)
#matrix = darts.sample(1)[0]
#print(matrix[0])
#print(matrix[1])


normal_matrix = np.array([
    [3, 3, 0, 0, 0],
    [0, 5, 2, 0, 0],
    [0, 0, 3, 1, 0],
    [0, 0, 1, 0, 7]
])

reduce_matrix = np.array([
    [2, 6, 0, 0, 0],
    [2, 0, 3, 0, 0],
    [5, 0, 5, 0, 0],
    [0, 7, 0, 0, 4]
])

print("Normal Matrix:")
print(normal_matrix)
print("\nReduce Matrix:")
print(reduce_matrix)


matrix = (normal_matrix, reduce_matrix)

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
