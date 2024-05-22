import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import BENCH_PRIMITIVES
from genotypes import Bench_Genotype

class MixedOp(nn.Module):

  def __init__(self, C_in, C_out, stride, affine, track_running_stats):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in BENCH_PRIMITIVES:
      op = OPS[primitive](C_in, C_out, stride, affine, track_running_stats)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class NASBench201Cell(nn.Module):

    def __init__(self, num_nodes, C_in, C_out, stride, bn_affine=False, track_running_stats=True):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = num_nodes # number of NODES
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.bn_affine = bn_affine

        #ORDERING [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)] archs of the DAG
        self._ops = nn.ModuleList()

        for i in range(self.NUM_NODES):
            for j in range(i):
                if j==0:
                    op = MixedOp(C_in, C_out, stride, bn_affine, track_running_stats)
                else:
                    op = MixedOp(C_in, C_out, 1, bn_affine, track_running_stats)
                self._ops.append(op)

    def forward(self, input, weights): 
        nodes = [input]
        offset = 0
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self._ops[offset + j](nodes[j], weights[offset + j]) for j in range(i))
            nodes.append(node_feature)
            offset += i
        return nodes[-1]

class BenchNetwork(nn.Module):

  def __init__(self, C, num_classes, stages, cells, criterion, steps=4):
    super(BenchNetwork, self).__init__()
    self._C = C #init number of channels
    self._num_classes = num_classes
    self._stages = stages #number of stages
    self._criterion = criterion
    self.NUM_NODES = steps #number of nodes per cell
    self._cells = cells #number of cells per stage
    self.norm_cells = nn.ModuleList() #list of normal cells
    self.red_cells = nn.ModuleList() #list of reduction cells
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
    C_curr = C
    for i in range(self._stages):
      self.norm_cells.append(self._make_stage(C_curr, self._cells))
      if i<self._stages-1:
        self.red_cells.append(ResNetBasicblock(C_curr, C_curr*2, stride=2))
        C_curr *= 2
    
    self.lastact = nn.Sequential(nn.BatchNorm2d(C_curr), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_curr, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = BenchNetwork(self._C, self._num_classes, self._stages, self._cells, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new
  
  def _make_stage(self, in_channels, num_cells):
        cells = []
        for i in range(num_cells):
            cells.append(NASBench201Cell(self.NUM_NODES, in_channels, in_channels, 1))
        return nn.Sequential(*cells)

  def forward(self, input):
    x = self.stem(input)
    weights = F.softmax(self.alphas_normal, dim=-1)
    for i in range(self._stages):
      for j in range(self._cells):
        x = self.norm_cells[i][j](x, weights) #cell j of stage i
      if i<self._stages-1:
        x = self.red_cells[i](x)
    x = self.lastact(x)
    x = self.global_pooling(x)
    logits = self.classifier(x.view(x.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = int(self.NUM_NODES * (self.NUM_NODES-1)/2) #sum(1 for i in range(self.NUM_NODES) for n in range(2+i))
    num_ops = len(BENCH_PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) #Tensor (archs, ops)
    self._arch_parameters = self.alphas_normal
  
  def _save_arch_parameters(self):
    self._saved_arch_parameters = self._arch_parameters.clone()
  
  def softmax_arch_parameters(self):
    self._save_arch_parameters()
    self._arch_parameters.data.copy_(F.softmax(self._arch_parameters, dim=-1))
          
  def restore_arch_parameters(self):
    self._arch_parameters.data.copy_(self._saved_arch_parameters)
    del self._saved_arch_parameters

  def arch_parameters(self):
    return [self._arch_parameters]
  
  def show_alphas(self):
    with torch.no_grad():
      return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self._arch_parameters, dim=-1).cpu())

  def genotype(self):

    def _parse(weights):
      gene = []
      start = 0
      for i in range(1, self.NUM_NODES):
          end = start + i
          W = weights[start:end].copy()
          for j in range(i):
              k_best = max(range(len(W[j])), key=lambda k: W[j][k])
              #To exclude 'none' in the choice:
              #k_best = max(range(len(W[j])), key=lambda k: W[j][k] if k != BENCH_PRIMITIVES.index('none') else -float('inf'))
              gene.append((BENCH_PRIMITIVES[k_best], j))
          start = end
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

    concat = range(int(self.NUM_NODES * (self.NUM_NODES-1)/2)) # number of archs in a DAG of NUM_NODES nodes
    genotype = Bench_Genotype(
        normal=gene_normal, normal_concat=concat
    )
    return genotype
  