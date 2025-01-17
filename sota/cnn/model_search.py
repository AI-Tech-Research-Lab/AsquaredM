import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import math

from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys
sys.path.insert(0, '../../')
from optimizers.darts.utils import drop_path

'''
class MixedOp(nn.Module):

    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
'''

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

beta_decay_scheduler = DecayScheduler(base_lr=1, 
                                            T_max=50, 
                                            T_start=0, 
                                            T_stop=50, 
                                            decay_type='linear')

class MixedOp(nn.Module):
  
  def __init__(self, C, stride, PRIMITIVES, auxiliary_skip=False, auxiliary_operation='skip'):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.stride = stride
    self.auxiliary_skip = auxiliary_skip

    if auxiliary_skip: #DARTS-: auxiliary skip connection
      if self.stride == 2:
        self.auxiliary_op = FactorizedReduce(C, C, affine=False)
      elif auxiliary_operation == 'skip':
        self.auxiliary_op = Identity()
      elif auxiliary_operation == 'conv1':
        self.auxiliary_op = nn.Conv2d(C, C, 1, padding=0, bias=False)
        # reinitialize with identity to be equivalent as skip
        # print(self.auxiliary_op.weight.data.size())
        eye = torch.eye(C,C)
        for i in range(C):
          self.auxiliary_op.weight.data[i,:,0,0] = eye[i]
        # print(self.auxiliary_op.weight.data)

      else:
        assert False, 'Unknown auxiliary operation'

    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    res = sum(w * op(x) for w, op in zip(weights, self._ops))
    if self.auxiliary_skip:
      res += self.auxiliary_op(x) * beta_decay_scheduler.decay_rate
    
    return res


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, auxiliary_skip):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives[edge_index], auxiliary_skip)
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights, drop_prob=0.):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if drop_prob > 0. and self.training:
                s = sum(drop_path(self._ops[offset+j](h, weights[offset+j]), drop_prob) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, primitives, steps=4,
                 multiplier=4, stem_multiplier=3, drop_path_prob=0.0, auxiliary_skip=False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = drop_path_prob

        nn.Module.PRIMITIVES = primitives

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, auxiliary_skip)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers,
                            self._criterion, self.PRIMITIVES,
                            drop_path_prob=self.drop_path_prob).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, updateType='alpha'):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if updateType == 'weight':
                    weights = self.alphas_reduce
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if updateType == 'weight':
                    weights = self.alphas_normal
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def _loss(self, input, target, updateType='alpha'):
        logits = self(input, updateType=updateType)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])

        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        

    def arch_parameters(self):
        return self._arch_parameters
  
    def _save_arch_parameters(self):
        self._saved_arch_parameters = [p.clone() for p in self._arch_parameters]
  
    def softmax_arch_parameters(self):
        self._save_arch_parameters()
        for p in self._arch_parameters:
            p.data.copy_(F.softmax(p, dim=-1))
            
    def restore_arch_parameters(self):
        for i, p in enumerate(self._arch_parameters):
            p.data.copy_(self._saved_arch_parameters[i])
        del self._saved_arch_parameters
  
    def clip(self):
        for p in self.arch_parameters():
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())

    def genotype(self):

        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError: # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[start+j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

