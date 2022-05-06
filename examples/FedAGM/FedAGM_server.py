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
        self.baseline_model = None
        # alpha controls the decreasing rate of the mapping function

    def get_trainer(self, model=None):
        return Trainer(model)

    async def aggregate_weights(self, updates):
        if self.delta is None:
            self.delta = self.algorithm.extract_weights()
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)

        baseline_model = self.algorithm.extract_weights()
        if self.delta is None:
            self.delta = {}
            for name in update.keys():
                self.delta[name] = torch.zeros(update[name].shape)
        if self.baseline_model is None:
            self.baseline_model = {}
            for name in update.keys():
                self.baseline_model[name] = baseline_model[name].detach().cpu()

        updated_weights = {}
        for name in update.keys():
            updated_weights[name] = self.tau * (update[name] + baseline_model[name]) + \
                                    (1 - self.tau) * (baseline_model[name])

        for name in update.keys():
            self.delta[name] = - (updated_weights[name] - self.baseline_model[name])
            self.baseline_model[name] = updated_weights[name]

        for name in update.keys():
            updated_weights[name] = updated_weights[name] - self.lambdaa * self.delta[name]

        self.algorithm.load_weights(updated_weights)

