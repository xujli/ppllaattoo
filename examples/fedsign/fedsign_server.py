"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import asyncio
from plato.config import Config

from plato.servers import fedavg
from plato.algorithms import registry as algorithms_registry
from plato.trainers import registry as trainers_registry
from fedsign_trainer import Trainer

import torch
import numpy as np

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

        # alpha controls the decreasing rate of the mapping function
        self.alpha = 5
        self.local_correlations = {}
        self.adaptive_weighting = None

        self.momentum_update_direction = None
        self.alpha = Config().trainer.learning_rate
        self.momentum = Config().trainer.momentum
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        if Config().data.partition_size % Config().trainer.batch_size != 0:
            self.batch_nums += 1

        self.coef1 = torch.sum(torch.pow(self.momentum, torch.arange(1, self.batch_nums + 1))) * self.alpha
        self.coef2 = 0
        for i in range(self.batch_nums):
            for j in range(0, i + 1):
                self.coef2 += self.momentum ** j
        self.coef2 = self.coef2 * self.alpha * (self.momentum - 1)

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
        num_samples = [report.num_samples for (report, __) in updates]
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])
        # Get adaptive weighting based on both node contribution and date size
        update_received = self.algorithm.compute_weight_updates(weights_received)
        self.adaptive_weighting = self.calc_contribution(update_received, num_samples)

        return update_received

    async def federated_averaging(self, updates):
        # update (report, (weights, ...))
        """ Aggregate weight updates and deltas updates from the clients. """
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])

        # Perform weighted averaging
        avg_updates = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_updates[name] += delta * self.adaptive_weighting[i]
            # Yield to other tasks in the server
            await asyncio.sleep(0)

        if self.momentum_update_direction is None:
            self.momentum_update_direction = {}
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                self.momentum_update_direction[name] = 1 / (1 - self.momentum) * (-delta) / (self.alpha * self.batch_nums)
        else:
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                self.momentum_update_direction[name] = self.momentum_update_direction[name] * self.momentum + \
                                                       ((-delta) / (self.alpha * self.batch_nums) - \
                                                        self.momentum * self.momentum_update_direction[name])

        return avg_updates

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.momentum_update_direction]

    def calc_contribution(self, updates, num_samples):
        """ Calculate the node contribution based on the angle between the local
        and global gradients. """
        correlations, contribs = [None] * len(updates), [None] * len(updates)

        # Perform weighted averaging
        avg_grad = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in updates[0].items()
        }

        for i, update in enumerate(updates):
            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_grad[name] += delta * (num_samples[i] / self.total_samples)

        # Update the baseline model weights
        curr_global_grads = self.process_grad(avg_grad)

        # Compute angles in radian between local and global gradients
        for i, update in enumerate(updates):
            local_grads = self.process_grad(update)
            inner = np.inner(curr_global_grads, local_grads)
            norms = np.linalg.norm(curr_global_grads) * np.linalg.norm(local_grads)
            correlations[i] = np.arccos(np.clip(inner / norms, -1.0, 1.0))
        # Calculate the weighting of each participating client for aggregation
        adaptive_weighting = [None] * len(updates)
        total_weight = 0.0
        for i, contrib in enumerate(correlations):
            total_weight += num_samples[i] * np.exp(contrib)
        for i, contrib in enumerate(correlations):
            adaptive_weighting[i] = (num_samples[i] * np.exp(contrib)) / total_weight
        return adaptive_weighting

    @staticmethod
    def process_grad(grads):
        """Convert gradients to a flattened 1-D array."""
        grads = list(dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for i in range(1, len(grads)):
            flattened = np.append(flattened, grads[i])

        return flattened