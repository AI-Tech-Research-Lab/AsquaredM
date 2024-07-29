from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = ['none', 'skip_connect', 'dua_sepc_3x3', 'dua_sepc_5x5', 'dil_sepc_3x3', 'dil_sepc_5x5', 'avg_pool_3x3', 'max_pool_3x3']

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[
        ('nor_conv_3x3', 0), ('nor_conv_3x3', 1), 
        ('nor_conv_3x3', 0), ('nor_conv_3x3', 1), 
        ('nor_conv_3x3', 1), ('skip_connect', 0), 
        ('skip_connect', 0), ('dil_sepc_3x3', 2)
    ], 
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1), 
        ('skip_connect', 2), ('max_pool_3x3', 1), 
        ('max_pool_3x3', 0), ('skip_connect', 2), 
        ('skip_connect', 2), ('max_pool_3x3', 1)
    ], 
    reduce_concat=[2, 3, 4, 5])
'''
Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
'''

DARTS = DARTS_V2

BETADARTS = Genotype(normal=[('dua_sepc_3x3', 1), ('dua_sepc_5x5', 0), ('dua_sepc_3x3', 0), ('dua_sepc_5x5', 2), ('dua_sepc_3x3', 3), ('dua_sepc_3x3', 2), ('dua_sepc_3x3', 3), ('dua_sepc_5x5', 2)], normal_concat=range(2, 6), reduce=[('dua_sepc_5x5', 0), ('dua_sepc_5x5', 1), ('dua_sepc_5x5', 2), ('max_pool_3x3', 0), ('dil_sepc_5x5', 3), ('dil_sepc_5x5', 2), ('dua_sepc_5x5', 2), ('dil_sepc_5x5', 3)], reduce_concat=range(2, 6))
DARTS = Genotype(normal=[('dua_sepc_3x3', 1), ('skip_connect', 0), ('dua_sepc_3x3', 0), ('skip_connect', 1), ('dua_sepc_3x3', 0), ('dua_sepc_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_sepc_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dua_sepc_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

# Convert Genotype to a serializable dictionary
def genotype_to_dict(genotype):
    # Create a dictionary with mandatory fields
    genotype_dict = {
        'normal': genotype.normal,
        'normal_concat': list(genotype.normal_concat)
    }
    
    # Check if 'reduce' and 'reduce_concat' attributes exist
    if hasattr(genotype, 'reduce'):
        genotype_dict['reduce'] = genotype.reduce
    if hasattr(genotype, 'reduce_concat'):
        genotype_dict['reduce_concat'] = list(genotype.reduce_concat)
    
    return genotype_dict

dict = genotype_to_dict(DARTS)

#save dict into a path dir
import json
import os
path = 'results/darts_search_datasetcifar10_samFalse_betadecayFalse'
with open(os.path.join(path, 'genotype.json'), 'w') as f:
    json.dump(dict, f)

