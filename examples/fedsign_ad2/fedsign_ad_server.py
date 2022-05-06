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
from fedsign_ad_trainer import Trainer

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

        # if self.client_momentum_update_direction is None:
        #     gradients_update = [{name: -delta / self.lr / self.batch_nums for name, delta in update.items()} for update in update_received]
        #
        # else:
        #     gradients_update = [{name: -delta / self.lr / self.batch_nums - self.client_momentum * self.client_momentum_update_direction[name] for name, delta in update.items()} for update
        #                         in update_received]
        # self.clients_gradient = gradients_update
        #
        # self.client_gradient = {}
        # for name in gradients_update[0].keys():
        #     self.client_gradient[name] = torch.zeros(gradients_update[0][name].shape)
        # for i, grad in enumerate(gradients_update):
        #     for name, delta in grad.items():
        #         self.client_gradient[name] += delta * num_samples[i] / self.total_samples
        # old_weights = self.algorithm.extract_weights()
        # self.norm_matrix, self.deviation_matrix = {}, {}
        # for name in old_weights.keys():
        #     layer_norm, layer_deviation = [], [] # num_selected_works ,num_selected_works
        #     if not self.old_grad is None:
        #         for update, weight, grad in zip(update_received, weights_received, self.old_grad):
        #             norm = np.linalg.norm(update[name].detach().cpu().numpy().flatten())
        #             deviation = np.inner(grad[name].detach().cpu().numpy().flatten(), weight[name].detach().cpu().numpy().flatten()) / \
        #                         (np.linalg.norm(grad[name].detach().cpu().numpy().flatten()) * np.linalg.norm(weight[name].detach().cpu().numpy().flatten()))
        #
        #             layer_norm.append(norm)
        #             layer_deviation.append(deviation)
        #     else:
        #         for update in update_received:
        #             norm = np.linalg.norm(update[name].detach().cpu().numpy().flatten(), 2)
        #             layer_norm.append(norm)
        #             layer_deviation.append(0)
        #
        #     self.norm_matrix[name] = layer_norm
        #     self.deviation_matrix[name] = layer_deviation

        # self.old_grad = gradients_update
        return update_received

    async def federated_averaging(self, updates):
        # update (report, (weights, ...))
        """ Aggregate weight updates and deltas updates from the clients. """
        update_received = self.extract_client_updates(updates)
        num_samples = [report.num_samples for (report, __) in updates]
        # Perform weighted averaging
        avg_updates = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in update_received[0].items()
        }
        # self.weight_matrix = []
        # for weights in weights_received:
        #     weight = []
        #     for name, delta in weights.items():
        #         weight.append(-np.linalg.norm(avg_updates[name] - delta))
        #     self.weight_matrix.append(np.exp(weight))
        # self.weight_matrix = self.weight_matrix / np.sum(self.weight_matrix, axis=0)
        # # Perform weighted averaging
        # avg_updates = {
        #     name: self.trainer.zeros(weights.shape)
        #     for name, weights in weights_received[0].items()
        # }

        # Use adaptive weighted average
        for i, update in enumerate(update_received):
            for j, (name, delta) in enumerate(update.items()):
                avg_updates[name] += delta * num_samples[i] / self.total_samples

        # avg_updates = await super(Server, self).federated_averaging(updates)
        if self.client_momentum_update_direction is None:
            self.client_momentum_update_direction = {}
            self.client_gradient = {}
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                    continue
                self.client_gradient[name] = (-delta) / (self.lr * self.batch_nums)
                self.client_momentum_update_direction[name] = self.client_gradient[name]
        else:
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                    continue
                self.client_gradient[name] = ((-delta) / (self.lr * self.batch_nums) - \
                                                        self.client_momentum * self.client_momentum_update_direction[name])
                self.client_momentum_update_direction[name] = self.client_momentum_update_direction[name] * self.client_momentum + \
                                                                self.client_gradient[name]


        return avg_updates
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
        return [payload, self.client_momentum_update_direction, self.current_round]

    def choose_clients(self, clients_pool, clients_count):
        """ Choose a subset of the clients to participate in each round. """
        assert clients_count <= len(clients_pool)
        if self.probabilities is not None:
        # Select clients randomly
            results = np.random.choice(np.arange(1, self.total_clients+1), clients_count, replace=False, \
                                    p=list(self.probabilities.values()) / np.sum(list(self.probabilities.values())))
            return [int(item) for item in results]
        else:
            self.results = random.sample(clients_pool, clients_count)

            return self.results

    #
    @staticmethod
    def process_grad(grads):
        """Convert gradients to a flattened 1-D array."""
        grads = list(dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for i in range(1, len(grads)):
            flattened = np.append(flattened, grads[i])

        return flattened
