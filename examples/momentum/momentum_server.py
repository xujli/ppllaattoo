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


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.momentum_update_direction = None
        self.momentum_direction_received = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload[0] for (__, payload) in updates]

        self.momentum_direction_received = [
            payload[1] for (__, payload) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """

        update = await super().federated_averaging(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])

        self.momentum_update_direction = self.momentum_direction_received[0]
        for model in range(len(self.momentum_update_direction)):
            for layer in range(len(self.momentum_update_direction[model])):
                self.momentum_update_direction[model][layer] *= updates[0][0].num_samples / self.total_samples

        # Update server momentum direction
        for client in range(1, len(self.momentum_direction_received)):
            for model in range(len(self.momentum_direction_received[client])):
                for layer in range(len(self.momentum_direction_received)):
                    self.momentum_update_direction[model][layer] += \
                        updates[client][0].num_samples / self.total_samples * \
                        self.momentum_direction_received[client][model][layer]


        return update

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.momentum_update_direction]
