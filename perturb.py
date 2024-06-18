import numpy as np
import copy
from train_utils import validate

def create_perturb(model, noise="mult"):
    z = []
    for p in model.parameters():
        r = p.clone().detach().normal_()
        if noise == "mult":
            z.append(p.data * r)  # multiplicative noise
        else:
            z.append(r)           # additive noise 
    return z
    
def perturb(model, z, noise_ampl):
    for i,p in enumerate(model.parameters()):
        p.data += noise_ampl * z[i]

def calculate_robustness(device, net, loader, sigma):
    M = 20
    robustness = []

    acc0 = validate(loader, net, device, print_info=False)

    for _ in range(M):
        perturbed_net = copy.deepcopy(net)
        z = create_perturb(perturbed_net)
        perturb(perturbed_net, z, sigma)
        acc = validate(loader, perturbed_net, device, print_info=False)
        rob = abs(acc0 - acc) #/ acc0
        print(f"\nacc0: {acc0}; acc: {acc}; robustness: {rob}")
        robustness.append(rob)
    return sum(robustness) / len(robustness)

def calculate_robustness_list(device, net, loader, sigma_list):
    rob_list=[]
    for sigma in sigma_list:
        rob_list.append(calculate_robustness(device, net, loader, sigma))
    return rob_list

def get_net_info_runtime(device, net, loader, sigma_list, print_info=False):

    # robustness
    #sigma = 0.05
    net_info={}
    net_info['robustness'] = calculate_robustness_list(device, net, loader, sigma_list)
    net_info['robustness'] = [np.round(x, 2) for x in net_info['robustness']]

    if print_info:
        # print(net)
        print('Robustness: ', (net_info['robustness']))

    return net_info