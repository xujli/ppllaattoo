"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import torch
from torch import optim

class FedSignOptimizer(optim.SGD):
    """A customized optimizer for SCAFFOLD."""
    def __init__(self,
                 params,
                 lr,
                 bias,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)
        self.bias = bias

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

            for p, bias in zip(
                    group['params'], self.bias.values()):
                if p.grad is None:
                    continue
                d_p = p.grad.data
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


                # Update weight
                d_p.add_(bias, alpha=1)
                p.data.add_(d_p, alpha=-group['lr'])

        return loss