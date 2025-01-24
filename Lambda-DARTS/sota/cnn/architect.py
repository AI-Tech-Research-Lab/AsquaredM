import math
import numpy as np
import torch
from torch.autograd import Variable, grad
import sys


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
        self.k=1
        self.rho_alpha=0.1
        self.epsilon=0.01
        self.sam=args.sam

    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self._train_loss(model=self.model, input=input, target=target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            if not self.sam:
                self._backward_step(input_valid, target_valid)
            else:
                self._backward_step_SAM(input_valid, target_valid)
        self.optimizer.step()
    
    def _backward_step(self, input_valid, target_valid):
        loss = self._val_loss(self.model, input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = self._val_loss(model=unrolled_model, input=input_valid, target=target_valid)

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # Compute expression (8) from paper
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

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
            p.data.add_(R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    
    def compute_alpha_tilde(self, dL_val_dalpha, input_val, target_val):
        
        #rho_alpha = 0.5
        #k=5
        #fast_alpha_size set by a MultiStepLR every 50 epochs [50,100,150]
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
        #lr=0.1

        if self.k>1: #lookbehind SAM
            fast_alpha_size = self.lr #self.scheduler.get_last_lr()[0] #self.fast_epsilon
            # Save current architecture parameters as the "slow weights"
            slow_alpha = [alpha.clone() for alpha in self.model.arch_parameters()]
            fast_alpha = [alpha.clone() for alpha in self.model.arch_parameters()]
            tilde_alpha = [alpha.clone() for alpha in self.model.arch_parameters()]
            dL_val_dtilde = dL_val_dalpha
            #fast_alpha_init = [alpha.clone() for alpha in self.model.arch_parameters()]
            #fast_alpha_first = [alpha.clone() for alpha in self.model.arch_parameters()]

            # Lookbehind-SAM: k-step perturbation
            for step in range(self.k):

                # Compute the perturbation (gradient ascent step)
                tilde_alpha = [
                    alpha + self.rho_alpha * dalpha
                    for alpha, dalpha in zip(tilde_alpha, dL_val_dtilde)
                ]

                with torch.no_grad():
                    for param, alpha_p in zip(self.model.arch_parameters(), tilde_alpha):
                        param.data.copy_(alpha_p)

                # Compute the perturbed loss and gradients
                if self.betadecay:
                    val_loss_tilde = self._val_beta_loss(self.model, input_val, target_val, epoch)
                else:
                    val_loss_tilde = self._val_loss(self.model, input_val, target_val)

                dL_val_dtilde = torch.autograd.grad(val_loss_tilde, self.model.arch_parameters())

                # Perform gradient descent (update fast weights)
                fast_alpha = [
                    alpha - fast_alpha_size * dalpha
                    for alpha, dalpha in zip(fast_alpha,dL_val_dtilde) 
                ]

                if self.sgd_alpha and step==1: #for adaptive alpha
                    fast_alpha_first = fast_alpha
                
            if self.sgd_alpha:
                # adaptive alpha according to gradients alignment
                diff1=[v1 - v for v1,v in zip(fast_alpha_first, slow_alpha)] #fast_alpha_first - slow_alpha #fast_alpha_init
                diffk=[vk - v for vk,v in zip(fast_alpha, slow_alpha)] #fast_alpha_init
                #theta = diff1*diffk/torch.norm(diff1,2)*torch.norm(diffk,2)
                diff1_tensor = torch.cat([d.view(-1) for d in diff1])
                diffk_tensor = torch.cat([d.view(-1) for d in diffk])
                theta = torch.nn.CosineSimilarity(dim=0)(diff1_tensor, diffk_tensor)
                slow_alpha_size = (math.cos(theta.item())+1)/2 #between 0 and 1
                self.update_arch_lr(slow_alpha_size)
            
            '''
            slow_alpha = [
                slow + slow_alpha_size * (fast - slow)
                for slow, fast in zip(slow_alpha, fast_alpha)
            ]
            '''
            
            #dL_val_dtilde = [fast - slow for slow, fast in zip(slow_alpha, fast_alpha)]
            dL_val_dtilde_alpha = [slow - fast for slow, fast in zip(slow_alpha, fast_alpha)] #reverts the gradient direction
        else:
            # Compute the perturbation (gradient ascent step)
            tilde_alpha = [
                alpha + self.rho_alpha * dalpha
                for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dalpha)
            ]

            # Copy the final perturbed weights back
            with torch.no_grad():
                for param, alpha_tilde in zip(self.model.arch_parameters(), tilde_alpha):
                    param.data.copy_(alpha_tilde)
            
            val_loss_tilde = self._val_loss(self.model, input_val, target_val)
            
            dL_val_dtilde_alpha = torch.autograd.grad(val_loss_tilde, self.model.arch_parameters())

        return dL_val_dtilde_alpha
    
    def _backward_step_SAM(self, input_val, target_val):

        val_loss = self._val_loss(self.model, input_val, target_val)
            
        dL_val_dalpha = torch.autograd.grad(val_loss, self.model.arch_parameters())

        #tilde_alpha = [alpha + self.rho_alpha * dalpha for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dalpha)]

        current_alpha = [alpha.clone() for alpha in self.model.arch_parameters()]

        dL_val_dtilde_alpha = self.compute_alpha_tilde(dL_val_dalpha, input_val, target_val)

        '''
        with torch.no_grad():
            for param, alpha_p in zip(self.model.arch_parameters(), tilde_alpha):
                param.data.copy_(alpha_p)
        
        
        if self.betadecay:
            val_loss_tilde = self._val_beta_loss(self.model, input_val, target_val, epoch)
        else:
            val_loss_tilde = self._val_loss(self.model, input_val, target_val)
        
        dL_val_dtilde_alpha = torch.autograd.grad(val_loss_tilde, self.model.arch_parameters())
        '''
        
        alpha_plus = [alpha + self.epsilon * dalpha for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dtilde_alpha)]
        alpha_minus = [alpha - self.epsilon * dalpha for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dtilde_alpha)]

        with torch.no_grad():
            for param, alpha_p in zip(self.model.arch_parameters(), alpha_plus):
                param.data.copy_(alpha_p)

        val_loss_plus = self._val_loss(self.model, input_val, target_val)

        dL_val_plus = torch.autograd.grad(val_loss_plus, self.model.arch_parameters())

        with torch.no_grad():
            for param, alpha_m in zip(self.model.arch_parameters(), alpha_minus):
                param.data.copy_(alpha_m)

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
