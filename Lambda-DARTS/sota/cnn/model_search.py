import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable

from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys
sys.path.insert(0, '../../')
from sota.cnn.utils import drop_path


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


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
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
                op = MixedOp(C, stride, self.primitives[edge_index])
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
                 multiplier=4, stem_multiplier=3, drop_path_prob=0.0, epsilon_0=0.001, lambda_=0.125, corr_type='none'):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = drop_path_prob

        self.corr_type = corr_type
        self.epsilon_0 = epsilon_0
        self.lambda_ = lambda_
        self.epsilon = 0.
        self.weights = {}

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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
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

    def forward(self, input, updateType='alpha', pert=None):
        s0 = s1 = self.stem(input)
        self.weights['normal'] = []
        self.weights['reduce'] = []
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if updateType == 'weight':
                    weights = self.alphas_reduce.clone()
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if updateType == 'weight':
                    weights = self.alphas_normal.clone()
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            if self.training:
                weights.retain_grad()
                self.weights['reduce' if cell.reduction else 'normal'].append(weights)
            if pert:
                weights = weights - pert[i]
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

    def get_arch_grads(self):
        grads_normal = [w.grad.data.clone().detach().reshape(-1) for w in self.weights['normal']]
        grads_reduce = [w.grad.data.clone().detach().reshape(-1) for w in self.weights['reduce']]
        return grads_normal, grads_reduce

    def get_perturbations(self):
        grads_normal, grads_reduce = self.get_arch_grads()

        def get_perturbation_for_cell(layer_gradients):
            with torch.no_grad():
                weight = 1 / ((len(layer_gradients) * (len(layer_gradients) - 1)) / 2)
                if self.corr_type == 'corr':
                    u = [g / g.norm(p=2.0) for g in layer_gradients]
                    sum_u = sum(u)
                    I = torch.eye(sum_u.shape[0]).cuda()
                    P = [(1 / g.norm(p=2.0)) * (I - torch.ger(u_l, u_l)) for g, u_l in zip(layer_gradients, u)]
                    perturbations = [weight * (P_l @ sum_u).reshape(self.alphas_normal.shape) for P_l in P]
                elif self.corr_type == 'signcorr':
                    perturbations = []
                    for i in range(len(layer_gradients)):
                        dir = 0
                        for j in range(len(layer_gradients)):
                            if i == j: continue
                            g, g_ = layer_gradients[i], layer_gradients[j]
                            dot, abs_dot = torch.dot(g, g_), torch.dot(torch.abs(g), torch.abs(g_))
                            dir += (torch.ones_like(g_) - (dot / abs_dot) * torch.sign(g) * torch.sign(g_)) * g_ / abs_dot
                        perturbations.append(weight * dir.reshape(self.alphas_normal.shape))
            return perturbations

        pert_normal = get_perturbation_for_cell(grads_normal)
        # pert_reduce = get_perturbation_for_cell(grads_reduce)
        pert_reduce = [torch.zeros_like(self.alphas_reduce) for _ in grads_reduce] # NO PERTURBATIONS ON REDUCTION CELLS
        self.epsilon = self.epsilon_0 / torch.cat(pert_normal + pert_reduce, dim=0).norm(p=2.0).item()

        idx_normal = 0
        idx_reduce = 0
        pert = []
        for cell in self.cells:
            if cell.reduction:
                pert.append(pert_reduce[idx_reduce] * self.epsilon)
                idx_reduce += 1
            else:
                pert.append(pert_normal[idx_normal] * self.epsilon)
                idx_normal += 1
        return pert

    def get_reg_grads(self, forward_grads, backward_grads):
        reg_grad = [(f - b).div_(2 * self.epsilon) for f, b in zip(forward_grads, backward_grads)]
        for idx, param in enumerate(self.parameters()):
            param.grad.data.add_(self.lambda_ * reg_grad[idx])

    def get_corr(self):
        grads_normal, grads_reduce = self.get_arch_grads()

        def corr(x):
            res = []
            norms = [x_.norm() for x_ in x]
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    res.append(
                        (torch.dot(x[i], x[j]) / (norms[i] * norms[j])).item())
            return sum(res) / len(res)
        return corr(grads_normal)
