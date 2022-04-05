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
from fedtrip_trainer import Trainer

import numpy as np
import random
import torch


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.client_gradient = None
        self.client_gradient_backup = None
        self.client_momentum_update_direction = None
        self.server_momentum_update_direction = None
        self.lr = Config().trainer.learning_rate
        self.client_momentum = Config().trainer.momentum
        self.server_momentum = Config().trainer.beta
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        if Config().data.partition_size % Config().trainer.batch_size != 0:
            self.batch_nums += 1
        self.batch_nums *= Config().trainer.epochs
        # alpha controls the decreasing rate of the mapping function
        self.moving_mean = {}
        self.alpha = 0.3
        self.local_correlations = {}
        self.clients_bias = None
        self.weight_matrix = []
        self.old_grad = None
        self.probabilities = None
        self.diff = {}
        self.diff_count = {}


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

    def extract_client_updates(self, updates):
        weights_received = [payload[0] for (__, payload) in updates]
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])
        # Get adaptive weighting based on both node contribution and date size
        update_received = self.algorithm.compute_weight_updates(weights_received)

        return update_received

    # async def federated_averaging(self, updates):
    #     # update (report, (weights, ...))
    #     """ Aggregate weight updates and deltas updates from the clients. """
    #     update_received = self.extract_client_updates(updates)
    #     num_samples = [report.num_samples for (report, __) in updates]
    #     # Perform weighted averaging
    #     avg_updates = {
    #         name: self.trainer.zeros(weights.shape)
    #         for name, weights in update_received[0].items()
    #     }
    #
    #
    #     # Use adaptive weighted average
    #     for i, update in enumerate(update_received):
    #         for j, (name, delta) in enumerate(update.items()):
    #             # avg_updates[name] += delta * (np.exp(-self.norm_matrix[name][i]) + self.alpha * np.exp(-self.deviation_matrix[name][i])) / \
    #             #                      (np.sum(np.exp(-np.array(self.norm_matrix[name]))) + self.alpha * np.sum(np.exp(-np.array(self.deviation_matrix[name]))))
    #             avg_updates[name] += delta * num_samples[i] / self.total_samples
    #
    #
    #     if self.client_momentum_update_direction is None:
    #         self.client_momentum_update_direction = {}
    #         self.client_gradient = {}
    #         # Use adaptive weighted average
    #         for name, delta in avg_updates.items():
    #             if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
    #                 continue
    #             self.client_gradient[name] = (-delta) / (self.lr * self.batch_nums)
    #             self.client_momentum_update_direction[name] = self.client_gradient[name]
    #     else:
    #         # Use adaptive weighted average
    #         for name, delta in avg_updates.items():
    #             if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
    #                 continue
    #             self.client_gradient[name] = ((-delta) / (self.lr * self.batch_nums) - \
    #                                                     self.client_momentum * self.client_momentum_update_direction[name])
    #             self.client_momentum_update_direction[name] = self.client_momentum_update_direction[name] * self.client_momentum + \
    #                                                             self.client_gradient[name]
    #
    #
    #
    #     return avg_updates
        # is reweighting useful?
        # moving mean
        # attentive weighting
        # direction and distance
        # STORM
        # VRLSGD
        # client selection

    # FLOB

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.current_round]

