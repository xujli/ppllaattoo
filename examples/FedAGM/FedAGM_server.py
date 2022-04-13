"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config
from plato.servers import fedavg
from FedAGM_trainer import Trainer
import torch

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.momentum_update_direction = None
        self.momentum_direction_received = None
        self.delta = None
        self.tau = Config().trainer.beta
        self.lambdaa = Config().trainer.lambdaa
        # alpha controls the decreasing rate of the mapping function

    def get_trainer(self, model=None):
        return Trainer(model)

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)

        baseline_weights = self.algorithm.extract_weights()
        if self.delta is None:
            self.delta = {}
            for name in update.keys():
                self.delta[name] = torch.zeros(update[name].shape)

        updated_weights = {}
        for name in update.keys():
            updated_weights[name] = self.tau * update[name] + \
                                    (1 - self.tau) * (baseline_weights[name] - self.lambdaa * self.delta[name])

        for name in update.keys():
            self.delta[name] = - (updated_weights[name] - baseline_weights[name])

        for name in update.keys():
            updated_weights[name] = updated_weights[name] - self.lambdaa * self.delta[name]

        self.algorithm.load_weights(updated_weights)

