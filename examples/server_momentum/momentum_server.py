"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config
from plato.servers import fedavg
import torch
import numpy as np
import math

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.momentum_update_direction = None
        self.momentum_direction_received = None

        # alpha controls the decreasing rate of the mapping function
        self.alpha = 5
        self.local_correlations = {}
        self.last_global_grads = None
        self.adaptive_weighting = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload[0] for (__, payload) in updates]

        self.momentum_direction_received = [
            payload[1] for (__, payload) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        # update (report, (weights, ...))
        """ Aggregate weight updates and deltas updates from the clients. """

        # Perform weighted averaging
        avg_update = await super().federated_averaging(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])
        num_samples = [report.num_samples for (report, __) in updates]

        self.momentum_update_direction = []

        for momentum in self.momentum_direction_received[0]:
            self.momentum_update_direction.append(
                [
                    self.trainer.zeros(weights.shape)
                    for weights in momentum
                ]
            )

        # Use adaptive weighted average
        for i, update in enumerate(self.momentum_direction_received):
            for idx1, momtentum in enumerate(update):
                for idx2, delta in enumerate(momtentum):
                    self.momentum_update_direction[idx1][idx2] += delta * num_samples[i] / self.total_samples

        return avg_update

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.momentum_update_direction]

