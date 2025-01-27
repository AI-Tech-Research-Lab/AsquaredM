import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import math, random
import warnings
import sys
sys.path.insert(0, '../')
from operations import ResNetBasicblock, OPS, NAS_BENCH_201, FactorizedReduce, Identity
from nasbench201.genotypes import Structure

class DecayScheduler(object):
    def __init__(self, base_lr=1.0, last_iter=-1, T_max=50, T_start=0, T_stop=50, decay_type='cosine'):
        self.base_lr = base_lr
        self.T_max = T_max
        self.T_start = T_start
        self.T_stop = T_stop
        self.cnt = 0
        self.decay_type = decay_type
        self.decay_rate = 1.0

    def step(self, epoch):
        if epoch >= self.T_start:
          if self.decay_type == "cosine":
              self.decay_rate = self.base_lr * (1 + math.cos(math.pi * epoch / (self.T_max - self.T_start))) / 2.0 if epoch <= self.T_stop else self.decay_rate
          elif self.decay_type == "slow_cosine":
              self.decay_rate = self.base_lr * math.cos((math.pi/2) * epoch / (self.T_max - self.T_start)) if epoch <= self.T_stop else self.decay_rate
          elif self.decay_type == "linear":
              self.decay_rate = self.base_lr * (self.T_max - epoch) / (self.T_max - self.T_start) if epoch <= self.T_stop else self.decay_rate
          else:
              self.decay_rate = self.base_lr
        else:
            self.decay_rate = self.base_lr

beta_decay_scheduler = DecayScheduler(base_lr=1.0, 
                                            T_max=50, 
                                            T_start=0, 
                                            T_stop=50, 
                                            decay_type='linear')

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class SearchCell(nn.Module):

  def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True, auxiliary_skip=False, auxiliary_operation='skip',
               forward_mode='default'):
    super(SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.edges     = nn.ModuleDict()
    self.max_nodes = max_nodes
    self.in_dim    = C_in #if forward_mode == 'default' else C_in // 4
    self.out_dim   = C_out #if forward_mode == 'default' else C_out // 4
    in_dim = self.in_dim if forward_mode == 'default' else self.in_dim // 4
    out_dim = self.out_dim if forward_mode == 'default' else self.out_dim // 4
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if j == 0:
          xlists = [OPS[op_name](in_dim, out_dim, stride, affine, track_running_stats) for op_name in op_names]
        else:
          xlists = [OPS[op_name](in_dim, out_dim,      1, affine, track_running_stats) for op_name in op_names]
        self.edges[ node_str ] = nn.ModuleList( xlists )
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)
    self.auxiliary_op=auxiliary_operation
    self.auxiliary_skip= auxiliary_skip
    self.forward_mode = forward_mode

    if auxiliary_skip: #DARTS-: auxiliary skip connection
      if stride == 2:
        self.auxiliary_op = FactorizedReduce(C_in, C_in, affine=False)
      elif auxiliary_operation == 'skip':
        self.auxiliary_op = Identity()
      elif auxiliary_operation == 'conv1':
        self.auxiliary_op = nn.Conv2d(C_in, C_in, 1, padding=0, bias=False)
        # reinitialize with identity to be equivalent as skip
        # print(self.auxiliary_op.weight.data.size())
        eye = torch.eye(C_in,C_in)
        for i in range(C_in):
          self.auxiliary_op.weight.data[i,:,0,0] = eye[i]
        # print(self.auxiliary_op.weight.data)

  def extra_repr(self):
    string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    return string
  
  def forward(self, inputs, weights):
    if self.forward_mode == 'pc':
        #print("Using PC forward")
        return self.forward_pc(inputs, weights)
    elif self.forward_mode == 'default':
        #print("Using default forward")
        return self.default_forward(inputs, weights)
    else:
        raise ValueError(f"Invalid forward_mode: {self.forward_mode}. Choose 'default' or 'pc'.")


  def default_forward(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        res = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) )
        if self.auxiliary_skip:  # Add auxiliary_op only if skip is True
          res += self.auxiliary_op(inputs) * beta_decay_scheduler.decay_rate
        inter_nodes.append(res)
      nodes.append( sum(inter_nodes))
    
    #res = nodes[-1]
    #if self.auxiliary_skip:
    #   res += self.auxiliary_op(inputs) * beta_decay_scheduler.decay_rate

    return nodes[-1]
  
  
  def forward_pc(self, x, weightss):
      # Channel proportion k=4
      dim_2 = x.shape[1]
      xtemp = x[:, :dim_2 // 4, :, :]   # First 1/4 channels
      xtemp2 = x[:, dim_2 // 4:, :, :]  # Remaining 3/4 channels

      nodes = [xtemp]
      for i in range(1, self.max_nodes):
          inter_nodes = []
          for j in range(i):
              node_str = '{:}<-{:}'.format(i, j)
              weights = weightss[self.edge2index[node_str]]
              res = sum(w * layer(nodes[j]) for w, layer in zip(weights, self.edges[node_str]))
              #if self.auxiliary_skip:  # Add auxiliary_op only if skip is True
              #    res += self.auxiliary_op(xtemp) * beta_decay_scheduler.decay_rate
              inter_nodes.append(res)
          nodes.append(sum(inter_nodes))

      # Combine final node with the remaining channels (xtemp2)
      temp1 = nodes[-1]
      #print("temp1 shape: ", temp1.shape)
      if temp1.shape[2] == x.shape[2]:  # Check spatial dimensions
          ans = torch.cat([temp1, xtemp2], dim=1)
      else:  # Apply pooling if dimensions mismatch
          ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
      #assert ans.shape[1] == dim_2, 'Invalid shape {:} vs {:}'.format(ans.shape, dim_2)
      # Perform channel shuffling
      ans = channel_shuffle(ans, 4)
      #print("ans shape: ", ans.shape)
      #assert ans.shape[1] == dim_2, 'Invalid shape {:} vs {:}'.format(ans.shape, dim_2)
      return ans
  
  '''
  def forward_pc(self, x, weightss):
    # Dynamically handle reduced channels: splitting inputs
    dim_2 = x.shape[1]
    xtemp = x[:, :dim_2 // 4, :, :]  # First 1/4 channels
    xtemp2 = x[:, dim_2 // 4:, :, :]  # Remaining 3/4 channels

    # Start processing with xtemp (1/4 channels)
    nodes = [xtemp]
    for i in range(1, self.max_nodes):
        inter_nodes = []
        for j in range(i):
            node_str = '{:}<-{:}'.format(i, j)
            weights = weightss[self.edge2index[node_str]]

            # Dynamically adjust the channel size of OPS based on the input size (xtemp here)
            #in_channels = xtemp.shape[1]  # Using xtemp only initially
            xlists = [OPS[op_name](dim_2//4, dim_2//4, 1, affine=True, track_running_stats=True) for op_name in self.op_names]
            res = sum(w * layer(nodes[j]) for w, layer in zip(weights, xlists))
            inter_nodes.append(res)

        nodes.append(sum(inter_nodes))

    # After processing nodes, combine the result with xtemp2 (remaining channels)
    temp1 = nodes[-1]
    if temp1.shape[2] == x.shape[2]:  # Check spatial dimensions
        ans = torch.cat([temp1, xtemp2], dim=1)  # Concatenate along the channel dimension
    else:  # Apply pooling if dimensions mismatch
        ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)

    # Perform channel shuffling
    ans = channel_shuffle(ans, 4)
    return ans
  '''

  # GDAS
  def forward_gdas(self, inputs, hardwts, index):
    nodes   = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = hardwts[ self.edge2index[node_str] ]
        argmaxs  = index[ self.edge2index[node_str] ].item()
        weigsum  = sum( weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie] for _ie, edge in enumerate(self.edges[node_str]) )
        inter_nodes.append( weigsum )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # joint
  def forward_joint(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        #aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
        aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) )
        inter_nodes.append( aggregation )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # uniform random sampling per iteration, SETN
  def forward_urs(self, inputs):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      while True: # to avoid select zero for all ops
        sops, has_non_zero = [], False
        for j in range(i):
          node_str   = '{:}<-{:}'.format(i, j)
          candidates = self.edges[node_str]
          select_op  = random.choice(candidates)
          sops.append( select_op )
          if not hasattr(select_op, 'is_zero') or select_op.is_zero is False: has_non_zero=True
        if has_non_zero: break
      inter_nodes = []
      for j, select_op in enumerate(sops):
        inter_nodes.append( select_op(nodes[j]) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # select the argmax
  def forward_select(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( self.edges[node_str][ weights.argmax().item() ]( nodes[j] ) )
        #inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # forward with a specific structure
  def forward_dynamic(self, inputs, structure):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      cur_op_node = structure.nodes[i-1]
      inter_nodes = []
      for op_name, j in cur_op_node:
        node_str = '{:}<-{:}'.format(i, j)
        op_index = self.op_names.index( op_name )
        inter_nodes.append( self.edges[node_str][op_index]( nodes[j] ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]


class Network(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space=NAS_BENCH_201, affine=False, track_running_stats=True, auxiliary_skip=False, 
               forward_mode='default'):
    super(Network, self).__init__()
    self._C        = C
    self._layerN   = N
    self._criterion = criterion
    self.max_nodes = max_nodes
    self.num_classes = num_classes
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats, auxiliary_skip=auxiliary_skip, forward_mode=forward_mode)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self._arch_parameters = nn.Parameter(1e-3*torch.randn(num_edge, len(search_space)) )

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def arch_parameters(self):
    return [self._arch_parameters]

  def show_alphas(self):
    with torch.no_grad():
      return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self._arch_parameters, dim=-1).cpu())

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self._arch_parameters[ self.edge2index[node_str] ]
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )
  
  def _loss(self, input, target, updateType='alpha'):
    logits = self(input, updateType=updateType)
    return self._criterion(logits, target)
  
  def binarization(self):
    self._save_arch_parameters()
    m, n = self._arch_parameters.size()
    maxIndexs = self._arch_parameters.data.cpu().numpy().argmax(axis=1)
    self._arch_parameters.data = self.proximal_step(self._arch_parameters, maxIndexs)
  
  def proximal_step(self, var, maxIndexs=None):
    values = var.data.cpu().numpy()
    m, n = values.shape
    alphas = []
    for i in range(m):
      for j in range(n):
        if j == maxIndexs[i]:
          alphas.append(values[i][j].copy())
          values[i][j] = 1
        else:
          values[i][j] = 0
    return torch.Tensor(values).cuda()
  
  def _save_arch_parameters(self):
    self._saved_arch_parameters = self._arch_parameters.clone()
  
  def softmax_arch_parameters(self):
    self._save_arch_parameters()
    self._arch_parameters.data.copy_(F.softmax(self._arch_parameters, dim=-1))
          
  def restore_arch_parameters(self):
    self._arch_parameters.data.copy_(self._saved_arch_parameters)
    del self._saved_arch_parameters

  def new(self):
        #(self, C, N, max_nodes, num_classes, criterion, search_space=NAS_BENCH_201, affine=False, track_running_stats=True):
        model_new = Network(self._C, self._layerN, self.max_nodes, self.num_classes, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

  def clip(self):
    for line in self._arch_parameters:
      max_index = line.argmax()
      line.data.clamp_(0, 1)
      if line.sum() == 0.0:
          line.data[max_index] = 1.0
      line.data.div_(line.sum())

  def forward(self, inputs, updateType='alpha'):
    if updateType == 'weight':
      alphas = self._arch_parameters
    else:
      alphas  = F.softmax(self._arch_parameters, dim=-1)

    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell(feature, alphas)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return logits
  

def distill(result):
  result = result.split('\n')
  cifar10 = result[5].replace(' ', '').split(':')
  cifar100 = result[7].replace(' ', '').split(':')
  imagenet16 = result[9].replace(' ', '').split(':')

  cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
  cifar10_test = float(cifar10[2][-7:-2].strip('='))
  cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
  cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
  cifar100_test = float(cifar100[3][-7:-2].strip('='))
  imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
  imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
  imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

  return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
    cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test




