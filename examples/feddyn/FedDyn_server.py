"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config
from plato.servers import fedavg
from FedDyn_trainer import Trainer
from plato.algorithms import registry as algorithms_registry
from plato.trainers import registry as trainers_registry
import torch
import numpy as np
import math

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.h = None
        self.coef = Config().clients.per_round / Config().clients.total_clients

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


        return avg_updates


    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)
        self.mu = Config().trainer.mu

        if self.h is None:
            self.h = {}
            for name, delta in update.items():
                self.h[name] = - self.mu * delta
        else:
            for name, delta in update.items():
                self.h[name] = self.h[name] - self.mu * delta * self.coef

        for name, delta in self.h.items():
            update[name] += - 1 / self.mu * delta
        updated_weights = self.algorithm.update_weights(update)

        self.algorithm.load_weights(updated_weights)