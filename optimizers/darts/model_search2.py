import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import ReLUConvBN, FactorizedReduce, OPS, DARTS_SPACE
from torch.autograd import Variable
from optimizers.darts.genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C_in, C_out, stride, affine, track_running_stats):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in DARTS_SPACE:
      op = OPS[primitive](C_in, C_out, stride, affine, track_running_stats)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine=False, track_running_stats=True):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      #self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine=False, track_running_stats=track_running_stats)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, 1, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, 1, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = op = MixedOp(C, C, stride, affine, track_running_stats) #MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, n_cells, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._n_cells = n_cells
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(n_cells):
      if i in [n_cells//3, 2*n_cells//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() ) + list( self.global_pooling.parameters() ) 
    xlist+= list( self.classifier.parameters() )
    return xlist
    #xlist+= list( self.lastact.parameters() ) 
  
  def show_alphas(self):
    with torch.no_grad():
      softmax_normal = nn.functional.softmax(self._arch_parameters[0], dim=-1).cpu()
      softmax_reduce = nn.functional.softmax(self._arch_parameters[1], dim=-1).cpu()
      return 'arch-parameters:\nNORMAL\n{:}\nREDUCE\n{:}'.format(softmax_normal, softmax_reduce)

  def new(self):
    model_new = Network(self._C, self._num_classes, self._n_cells, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, updateType='alpha'):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(DARTS_SPACE)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]
  
  def _save_arch_parameters(self):
        self._saved_arch_parameters = [p.clone() for p in self._arch_parameters]
    
  def _save_parameters(self):
      self._saved_parameters = [p.clone() for p in self.parameters()]
      
  def arch_parameters(self):
      return self._arch_parameters

  def softmax_arch_parameters(self, save=True):
      if save:
          self._save_arch_parameters()
      for p in self._arch_parameters:
          p.data.copy_(F.softmax(p, dim=-1))
          
  def restore_arch_parameters(self):
      for i, p in enumerate(self._arch_parameters):
          p.data.copy_(self._saved_arch_parameters[i])
      del self._saved_arch_parameters

  def restore_parameters(self):
      for i, p in enumerate(self.parameters()):
          p.data.copy_(self._saved_parameters[i])
      del self._saved_parameters

  def clip(self):
      for p in self.arch_parameters():
          for line in p:
              max_index = line.argmax()
              line.data.clamp_(0, 1)
              if line.sum() == 0.0:
                  line.data[max_index] = 1.0
              line.data.div_(line.sum())

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != DARTS_SPACE.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != DARTS_SPACE.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((DARTS_SPACE[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
