import torch
import json
import os
from typing import List, Text
import numpy as np
from genotypes import PRIMITIVES

class DARTS():

    def __init__(self,topk):
        self.input_nodes=2
        self.output_node=1
        self.topk=topk
        self.operations = len(PRIMITIVES)
        self.int_nodes = 4
        n_archs=0
        for i in range(0,self.int_nodes):
            n_archs+=(self.input_nodes+i) 
        self.n_archs=n_archs

    def sample(self, n_samples):
        archs = []
        matrix_size = self.input_nodes + self.int_nodes - 1
        for _ in range(n_samples):

            # create a random adjacency matrix compliant with DARTS rules

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
            archs.append(adjacency_matrix)

        return archs
    
    def create_neighboring_matrix(self, adjacency_matrix):
        int_nodes, matrix_size = adjacency_matrix.shape
        neighboring_matrix = np.copy(adjacency_matrix)
        # choose a random int node
        i = np.random.randint(0, int_nodes)
        # Choose a random edge from the selected node to alter
        edge_to_alter = np.random.randint(0, self.input_nodes + i)
        # If the edge is already present, change its operation
        if neighboring_matrix[i, edge_to_alter] != 0:
            new_operation = np.random.randint(1, self.operations)
            neighboring_matrix[i, edge_to_alter] = new_operation
        else:
            # Choose a random operation for the new edge
            new_operation = np.random.randint(1, self.operations)
            neighboring_matrix[i, edge_to_alter] = new_operation
            # Remove a random edge of the same node
            edge_to_zeroize = np.random.randint(0, self.input_nodes + i)
            while edge_to_zeroize == edge_to_alter:
                edge_to_zeroize = np.random.randint(0, self.input_nodes + i)
            neighboring_matrix[i, edge_to_zeroize] = 0 # Zeroize the edge

        return neighboring_matrix
    
    def encode_adjacency_matrix(self, adjacency_matrix):
        # Len encoding = num_int_nodes(num_int_nodes+5)/2 + input_nodes
        num_intermediate_nodes = adjacency_matrix.shape[0]
        op_input1= adjacency_matrix[0][0]
        op_input2= adjacency_matrix[0][1]
        ops_vector = [op_input1, op_input2]
        dag_vector = []
        num_input_nodes = self.input_nodes
        for i in range(1,num_intermediate_nodes):
                # Intermediate nodes
                preceding_nodes = np.nonzero(adjacency_matrix[i])[0]
                # Sort preceding nodes
                preceding_nodes.sort()
                # Create the encoding
                for node in range(i+num_input_nodes):
                    if node in preceding_nodes:
                        dag_vector.extend([1])
                    else:
                        dag_vector.extend([0])
                for node in preceding_nodes:
                    ops_vector.extend([adjacency_matrix[i][node]])  # Add a non-zero value and operation
        return dag_vector + ops_vector

    def decode_to_adjacency_matrix(self, encoded_vector):
        num_input_nodes = self.input_nodes
        num_intermediate_nodes = self.int_nodes 
        num_ops = 2 * num_intermediate_nodes
        offset = len(encoded_vector) - num_ops + 2
        
        # Initialize the adjacency matrix
        adjacency_matrix = np.zeros((num_intermediate_nodes, num_input_nodes + num_intermediate_nodes - 1), dtype=int)
        
        # Fill in the adjacency matrix using the encoded vector
        idx = 0
        j=0
        for i in range(num_intermediate_nodes):
            # Add connections to input nodes for the first intermediate node
            if i == 0:
                adjacency_matrix[i][:num_input_nodes] = encoded_vector[-num_ops:-num_ops+2]
            else:
                # Find the positions of 1s in the encoding indicating connections
                preceding_nodes = i + num_input_nodes
                connection_positions = [j for j, val in enumerate(encoded_vector[idx:idx+preceding_nodes]) if val == 1]
                # Fill in the adjacency matrix based on the connections
                for pos in connection_positions:
                    adjacency_matrix[i][pos] = encoded_vector[offset+j]
                    j+=1
                idx += preceding_nodes
        
        return adjacency_matrix
   

darts = DARTS(2)
matrix = darts.sample(1)[0]
print(matrix)
vector = darts.encode_adjacency_matrix(matrix)
print(vector)
matrix = darts.decode_to_adjacency_matrix(vector)
print(matrix)
