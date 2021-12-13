"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import torch
from torch import optim


class ScaffoldOptimizer(optim.SGD):
    """A customized optimizer for SCAFFOLD."""
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 alpha=1.):
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)
        self.server_update_direction = None
        self.client_id = None
        self.batches_num = 100
        self.interval = 0
        self.alpha = alpha

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # Initialize server update direction and client update direction
            if self.server_update_direction is None:
                self.server_update_direction = [0] * len(group['params'])
            idx = 0
            for p, server_update_direction in zip(
                    group['params'], self.server_update_direction):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                server_update_direction = server_update_direction.to(p.device)
                param_state = self.state[p]

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # Apply variance reduction
                d_p_bak = torch.clone(d_p).detach()
                d_p.add_(server_update_direction, alpha=1 / group['lr'] / self.batches_num * self.alpha)
                self.server_update_direction[idx] = server_update_direction * (1 - (1 / self.batches_num)) * (1 / self.interval) +\
                                                    d_p_bak * (1 / self.batches_num) * group['lr'] * (1 - 1 / self.interval)
                idx += 1

                # Update weight
                p.data.add_(d_p, alpha=-group['lr'])

        return loss