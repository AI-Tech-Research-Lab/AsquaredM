import torch
import math
from torch.autograd import Variable

class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay,
        )

        self._init_arch_parameters = []
        for alpha in self.model.arch_parameters():
            alpha_init = torch.zeros_like(alpha)
            alpha_init.data.copy_(alpha)
            self._init_arch_parameters.append(alpha_init)

        # Set mode based on method
        if args.method in ['darts', 'darts-proj','sdarts','sdarts-proj']:
            self.method = 'fo'
        elif args.method in ['darts-sam', 'darts-proj-sam']:
            self.method = 'darts-sam'
            self.rho_alpha = args.rho_alpha
            self.epsilon = 0.01
        elif 'so' in args.method:
            print('ERROR: PLEASE USE architect.py for second order darts')
        elif args.method in ['blank', 'blank-proj']:
            self.method = 'blank'
        else:
            print('ERROR: WRONG ARCH UPDATE METHOD', args.method); exit(0)

    def reset_arch_parameters(self):
        for alpha, alpha_init in zip(self.model.arch_parameters(), self._init_arch_parameters):
            alpha.data.copy_(alpha_init.data)

    def step(self, input_train, target_train, input_valid, target_valid, *args, **kwargs):
        if self.method == 'fo':
            shared = self._step_fo(input_valid, target_valid)
        elif self.method == 'darts-sam':
            shared = self._step_sam(input_valid, target_valid)
        elif self.method == 'so':
            raise NotImplementedError
        elif self.method == 'blank': ## do not update alpha
            shared = None
        return shared

    # First-order DARTS step
    def _step_fo(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
        self.optimizer.step()
        return None

    # DARTS-SAM step
    def _step_sam(self, input_valid, target_valid):
        # Compute gradients on the validation loss
        val_loss = self.model._loss(input_valid, target_valid)
        dL_val_dalpha = torch.autograd.grad(val_loss, self.model.arch_parameters(), create_graph=True)

        # Compute perturbed alpha (tilde_alpha)
        tilde_alpha = [
            alpha + self.rho_alpha * dalpha
            for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dalpha)
        ]
        with torch.no_grad():
            for param, alpha_tilde in zip(self.model.arch_parameters(), tilde_alpha):
                param.data.copy_(alpha_tilde)

        # Compute perturbed validation loss and gradients
        perturbed_loss = self.model._loss(input_valid, target_valid)
        dL_val_dtilde_alpha = torch.autograd.grad(perturbed_loss, self.model.arch_parameters(), create_graph=True)

        # Restore original alpha
        with torch.no_grad():
            for param, alpha_init in zip(self.model.arch_parameters(), self._init_arch_parameters):
                param.data.copy_(alpha_init)

        # Compute finite differences
        alpha_plus = [
            alpha + self.epsilon * dalpha
            for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dtilde_alpha)
        ]
        alpha_minus = [
            alpha - self.epsilon * dalpha
            for alpha, dalpha in zip(self.model.arch_parameters(), dL_val_dtilde_alpha)
        ]

        with torch.no_grad():
            for param, alpha_p in zip(self.model.arch_parameters(), alpha_plus):
                param.data.copy_(alpha_p)
        val_loss_plus = self.model._loss(input_valid, target_valid)

        with torch.no_grad():
            for param, alpha_m in zip(self.model.arch_parameters(), alpha_minus):
                param.data.copy_(alpha_m)
        val_loss_minus = self.model._loss(input_valid, target_valid)

        with torch.no_grad():
            for param, alpha_c in zip(self.model.arch_parameters(), self._init_arch_parameters):
                param.data.copy_(alpha_c)

        finite_diff = [
            (plus - minus).div(2 * self.epsilon)
            for plus, minus in zip(
                torch.autograd.grad(val_loss_plus, self.model.arch_parameters()),
                torch.autograd.grad(val_loss_minus, self.model.arch_parameters())
            )
        ]

        # Update architecture gradients
        final_grad = [
            dalpha + self.rho_alpha * fd
            for dalpha, fd in zip(dL_val_dalpha, finite_diff)
        ]
        for alpha, grad in zip(self.model.arch_parameters(), final_grad):
            if alpha.grad is None:
                alpha.grad = Variable(grad.data)
            else:
                alpha.grad.data.copy_(grad.data)

        # Update alpha
        self.optimizer.step()
        return None
