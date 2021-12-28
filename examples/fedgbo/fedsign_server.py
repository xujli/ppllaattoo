"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config

from plato.servers import fedavg
from plato.algorithms import registry as algorithms_registry
from plato.trainers import registry as trainers_registry
from fedsign_trainer import Trainer

import torch

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.momentum_update_direction = None
        self.alpha = Config().trainer.learning_rate
        self.momentum = Config().trainer.momentum
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        if Config().data.partition_size % Config().trainer.batch_size != 0:
            self.batch_nums += 1

    def get_trainer(self, model=None):
        return Trainer(model)

    def load_trainer(self):
        """Setting up the global model to be trained via federated learning."""

        get_trainer = getattr(self, 'get_trainer', trainers_registry.get)
        if self.trainer is None:
            self.trainer = get_trainer(self.model)

        self.trainer.set_client_id(client_id=0)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)

    async def federated_averaging(self, updates):
        # update (report, (weights, ...))
        """ Aggregate weight updates and deltas updates from the clients. """

        # Perform weighted averaging
        avg_updates = await super(Server, self).federated_averaging(updates)

        if self.momentum_update_direction is None:
            self.momentum_update_direction = {}
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                    continue
                self.momentum_update_direction[name] = (-delta) / (self.alpha * self.batch_nums)
        else:
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                    continue
                self.momentum_update_direction[name] = self.momentum_update_direction[name] * self.momentum + \
                                                       ((-delta) / (self.alpha * self.batch_nums) - \
                                                        self.momentum * self.momentum_update_direction[name])

        return avg_updates

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.momentum_update_direction]


