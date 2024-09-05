import logging
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.sam = args.sam
        self.rho_alpha = args.rho_alpha_sam
        self.epsilon = args.epsilon_sam
        self.betadecay = args.betadecay

    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)
    
    def _val_beta_loss(self, model, input, target, epoch):
        weights = 0 + 50 * epoch / 100  
        ssr_normal = self.mlc_loss(model._arch_parameters)
        loss = model._loss(input, target) + weights * ssr_normal
        return loss
        
    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self._train_loss(model=self.model, input=input, target=target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta - eta * (moment + dtheta))
        return unrolled_model
    
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch, unrolled):
        self.optimizer.zero_grad()
        if self.sam:
            self._backward_step_SAM(input_valid, target_valid, epoch)
        else:
            if unrolled: 
                self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
            else:
                self._backward_step(input_valid, target_valid, epoch)
        self.optimizer.step()

    def zero_hot(self, norm_weights):
        valid_loss = torch.log(norm_weights)
        base_entropy = torch.log(torch.tensor(2).float())
        aux_loss = torch.mean(valid_loss) + base_entropy
        return aux_loss
    
    def mlc_loss(self, arch_param):

        '''
        if isinstance(arch_param, list):
            # Concatenate matrices along dimension 0
            arch_param_concat = torch.cat(arch_param, dim=0)
        else:
            # Handle the case where arch_param is not a list
            arch_param_concat = arch_param
        '''

        # Compute the negative log-likelihood loss
        neg_loss = torch.logsumexp(arch_param, dim=-1)
        aux_loss = torch.mean(neg_loss)
        
        return aux_loss

    def mlc_loss_1setarchparams(self, arch_param):
        y_pred_neg = arch_param
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        aux_loss = torch.mean(neg_loss)
        return aux_loss

    def mlc_pos_loss(self, arch_param):
        act_param = F.softmax(arch_param, dim=-1)
        thr = act_param.max(dim=-1, keepdim=True)[0]
        y_true = (act_param >= thr)
        arch_param_new = (1 - 2 * y_true) * arch_param
        y_pred_neg = arch_param_new - y_true * 1e12
        y_pred_pos = arch_param_new - ~y_true * 1e12
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        aux_loss = torch.mean(neg_loss) + torch.mean(pos_loss)
        return aux_loss

    def mlc_loss2(self, arch_param):
        y_pred_neg = arch_param
        neg_loss = torch.log(torch.exp(y_pred_neg))
        aux_loss = torch.mean(neg_loss)
        return aux_loss

    def _backward_step(self, input_valid, target_valid, epoch):
        if self.betadecay:  # Beta-DARTS
            weights = 0 + 50 * epoch / 100
            if isinstance(self.model._arch_parameters, list): #DARTS
                # STEP DARTS 
                ssr_reduce = self.mlc_loss(self.model.alphas_reduce)
                ssr_normal = self.mlc_loss(self.model.alphas_normal)
                loss = self.model._loss(input_valid, target_valid) + weights*ssr_reduce + weights*ssr_normal
            else:
                ssr_normal = self.mlc_loss(self.model._arch_parameters)
                loss = self.model._loss(input_valid, target_valid) + weights * ssr_normal
        else:  # original DARTS
            loss = self.model._loss(input_valid, target_valid)        
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = self._val_loss(model=unrolled_model, input=input_valid, target=target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta * ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R * v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R * v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R * v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    
    def _backward_step_SAM(self, input_val, target_val, epoch):

        if self.betadecay:
            val_loss = self._val_beta_loss(self.model, input_val, target_val, epoch)
        else:
            val_loss = self._val_loss(self.model, input_val, target_val)
            
        dL_val_dalpha = torch.autograd.grad(val_loss, self.model.arch_parameters())

        tilde_alpha = [alpha + self.rho_alpha * dalpha for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dalpha)]

        current_alpha = [alpha.clone() for alpha in self.model.arch_parameters()]

        with torch.no_grad():
            for param, alpha_p in zip(self.model.arch_parameters(), tilde_alpha):
                param.data.copy_(alpha_p)

        if self.betadecay:
            val_loss_tilde = self._val_beta_loss(self.model, input_val, target_val, epoch)
        else:
            val_loss_tilde = self._val_loss(self.model, input_val, target_val)
        
        dL_val_dtilde_alpha = torch.autograd.grad(val_loss_tilde, self.model.arch_parameters())

        alpha_plus = [alpha + self.epsilon * dalpha for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dtilde_alpha)]
        alpha_minus = [alpha - self.epsilon * dalpha for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dtilde_alpha)]

        with torch.no_grad():
            for param, alpha_p in zip(self.model.arch_parameters(), alpha_plus):
                param.data.copy_(alpha_p)

        if self.betadecay:
            val_loss_plus = self._val_beta_loss(self.model, input_val, target_val, epoch)
        else:
            val_loss_plus = self._val_loss(self.model, input_val, target_val)

        dL_val_plus = torch.autograd.grad(val_loss_plus, self.model.arch_parameters())

        with torch.no_grad():
            for param, alpha_m in zip(self.model.arch_parameters(), alpha_minus):
                param.data.copy_(alpha_m)

        if self.betadecay:
            val_loss_minus = self._val_beta_loss(self.model, input_val, target_val, epoch)
        else:
            val_loss_minus = self._val_loss(self.model, input_val, target_val)

        dL_val_minus = torch.autograd.grad(val_loss_minus, self.model.arch_parameters())

        with torch.no_grad():
            for param, alpha_c in zip(self.model.arch_parameters(), current_alpha):
                param.data.copy_(alpha_c)

        finite_diff = [(plus - minus).div_(2 * self.epsilon) for plus, minus in zip(dL_val_plus, dL_val_minus)]

        first_order_approx = [dL_val_dalpha + self.rho_alpha * fd for dL_val_dalpha, fd in zip(dL_val_dtilde_alpha, finite_diff)]

        for v, g in zip(self.model.arch_parameters(), first_order_approx):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
