"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config

import torch
import numpy as np
from plato.servers import fedavg
from fedtrip_trainer import Trainer
from plato.servers.fedavg import *

from scipy.stats import wasserstein_distance


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.lr = Config().trainer.learning_rate
        self.client_momentum = Config().trainer.momentum
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        if Config().data.partition_size % Config().trainer.batch_size != 0:
            self.batch_nums += 1
        self.batch_nums *= Config().trainer.epochs
        self.EMD_distance = {}

    def get_trainer(self, model=None):
        return Trainer(model)

    def init_trainer(self):
        """Setting up the global model to be trained via federated learning."""

        get_trainer = getattr(self, 'get_trainer', trainers_registry.get)
        if self.trainer is None:
            self.trainer = get_trainer(self.model)

        self.trainer.set_client_id(client_id=0)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)

        for i in range(1, Config().clients.total_clients + 1):
            self.EMD_distance[i] = 0

    def compute_weight_deltas(self, updates):
        weights_received = [payload[0] for (__, __, payload, __) in updates]
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])
        # Get adaptive weighting based on both node contribution and date size
        update_received = self.algorithm.compute_weight_deltas(weights_received)

        flatten_update = []
        for params in self.algorithm.extract_weights().values():
            flatten_update.extend(np.reshape(params, -1))

        select_ids = [id for (id, __, __, __) in updates]
        for id, update in zip(select_ids, weights_received):
            flattened = []
            for params in update.values():
                flattened.extend(torch.flatten(params).numpy())

            if self.EMD_distance[id] == 0:
                self.EMD_distance[id] = wasserstein_distance(flattened, flatten_update)
            else:
                self.EMD_distance[id] = 0.5 * wasserstein_distance(flattened, flatten_update) + \
                    0.5 * self.EMD_distance[id]

        return update_received

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.current_round, self.EMD_distance, self.selected_clients]

