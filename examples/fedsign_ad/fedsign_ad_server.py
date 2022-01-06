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

import torch
import asyncio
import numpy as np

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
        self.alpha = 0.5
        self.local_correlations = {}
        self.weight_matrix = []


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
        # else:
        #     gradients_update = [{name: -delta / self.lr / self.batch_nums - self.client_momentum * self.client_momentum_update_direction[name] for name, delta in update.items()} for update
        #                         in update_received]
        old_weights = self.algorithm.extract_weights()

        self.weight_matrix = []
        for name in weights_received[0].keys():
            norm = []
            for weight in weights_received:
                norm.append(np.linalg.norm(old_weights[name] - weight[name]))
            mean = np.mean(norm)
            var = np.var(norm)
            weights = np.exp(-((norm-mean)**2/(2*var))) * 1 / (np.sqrt(var*2*np.pi))
            weights = weights / np.sum(weights)
            self.weight_matrix.append(weights)

        # losses = [payload[1] for (__, payload) in updates]
        # self.losses = np.sum(losses, axis=1)
        # self.weight = np.exp(-self.losses) / np.sum(np.exp(-self.losses))
        # print(self.weight)
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

        # Use adaptive weighted average
        # for i, update in enumerate(weights_received):
        #     for j, (name, delta) in enumerate(update.items()):
        #         avg_updates[name] += delta * num_samples[i] / self.total_samples
        #
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
                avg_updates[name] += delta * self.weight_matrix[j][i]

        # self.client_gradient = {
        #     name: self.trainer.zeros(weights.shape)
        #     for name, weights in weights_received[0].items()
        # }
        # for i, grad in enumerate(self.gradients_received):
        #     for name, delta in grad.items():
        #         self.client_gradient[name] += delta * norms[i][0] / sum(norms[i]) * num_samples[i] / self.total_samples
        # Perform weighted averaging
        # if self.server_momentum_update_direction is None:
        #     self.server_momentum_update_direction = {}
        #     # Use adaptive weighted average
        #     for name, delta in avg_updates.items():
        #         if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
        #             continue
        #         self.server_momentum_update_direction[name] = (-delta) / (self.lr * self.batch_nums)
        # else:
        #     # Use adaptive weighted average
        #     for name, delta in avg_updates.items():
        #         if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
        #             continue
        #         self.server_momentum_update_direction[name] = self.server_momentum_update_direction[name] * self.server_momentum + \
        #                                                ((-delta) / (self.lr * self.batch_nums) - self.coef / self.batch_nums * \
        #                                                 self.server_momentum * self.server_momentum_update_direction[name])

        # avg_updates = await super(Server, self).federated_averaging(updates)
        if self.client_momentum_update_direction is None:
            self.client_momentum_update_direction = {}
            self.client_gradient = {}
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                    continue
                self.client_gradient[name] = (-delta) / (self.lr * self.batch_nums)
                self.client_momentum_update_direction[name] = (-delta) / (self.lr * self.batch_nums)
                self.moving_mean[name] = delta
        else:
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                    continue
                self.client_gradient[name] = ((-delta) / (self.lr * self.batch_nums) - \
                                                        self.client_momentum * self.client_momentum_update_direction[name])
                self.client_momentum_update_direction[name] = self.client_momentum_update_direction[name] * self.server_momentum + \
                                                                self.client_gradient[name]

                self.moving_mean[name] = self.alpha * self.moving_mean[name] + 1 * delta
                avg_updates[name] = self.moving_mean[name]
        # if (len(self.loss_list) > 5) and (np.min(self.loss_list[:-5]) < self.loss):
        #     self.alpha /= 0.5
        # if (len(self.loss_list) > 5) and (np.max(self.loss_list[:-5]) > self.loss):
        #     self.alpha *= 0.5
        # if self.current_round < len(self.warmup):
        #     self.alpha = self.warmup[self.current_round]


        # if self.server_momentum_update_direction is None:
        #     self.server_momentum_update_direction = {}
        #     # Use adaptive weighted average
        #     for name, delta in avg_updates.items():
        #         if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
        #             continue
        #         self.server_momentum_update_direction[name] = self.client_gradient[name]
        # else:
        #     # Use adaptive weighted average
        #     for name, delta in avg_updates.items():
        #         if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
        #             continue
        #         self.server_momentum_update_direction[name] = self.server_momentum_update_direction[name] * \
        #                                                       self.server_momentum + self.client_gradient[name]

        # for name, delta in avg_updates.items():
        #     if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
        #         continue
        #     avg_updates[name] -= self.client_momentum_update_direction[name] * self.lr

        return avg_updates

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.client_momentum_update_direction]

    # def calc_adaptive_weighting(self, updates, num_samples):
    #     """ Compute the weights for model aggregation considering both node contribution
    #     and data size. """
    #     # Get the node contribution
    #     contribs = self.calc_contribution(updates, num_samples)
    #
    #     # Calculate the weighting of each participating client for aggregation
    #     adaptive_weighting = [None] * len(updates)
    #     total_weight = 0.0
    #     for i, contrib in enumerate(contribs):
    #         total_weight += num_samples[i] * np.exp(contrib)
    #     for i, contrib in enumerate(contribs):
    #         adaptive_weighting[i] = (num_samples[i] * np.exp(contrib)) / total_weight
    #
    #     return adaptive_weighting
    #
    # def calc_contribution(self, updates, num_samples):
    #     """ Calculate the node contribution based on the angle between the local
    #     and global gradients. """
    #     correlations, contribs = [None] * len(updates), [None] * len(updates)
    #
    #     # Perform weighted averaging
    #     avg_grad = {
    #         name: self.trainer.zeros(weights.shape)
    #         for name, weights in updates[0].items()
    #     }
    #
    #     for i, update in enumerate(updates):
    #         for name, delta in update.items():
    #             # Use weighted average by the number of samples
    #             avg_grad[name] += delta * (num_samples[i] / self.total_samples)
    #
    #     # Update the baseline model weights
    #     curr_global_grads = self.process_grad(avg_grad)
    #     distance = []
    #     # Compute angles in radian between local and global gradients
    #     for i, update in enumerate(updates):
    #         local_grads = self.process_grad(update)
    #         inner = np.inner(curr_global_grads, local_grads)
    #         norms = np.linalg.norm(curr_global_grads) * np.linalg.norm(local_grads)
    #         correlations[i] = np.arccos(np.clip(inner / norms, -1.0, 1.0))
    #
    #     for i, correlation in enumerate(correlations):
    #         client_id = self.selected_clients[i]
    #
    #         # Update the smoothed angle for all clients
    #         if client_id not in self.local_correlations.keys():
    #             self.local_correlations[client_id] = correlation
    #         self.local_correlations[client_id] = ((self.current_round - 1)
    #         / self.current_round) * self.local_correlations[client_id]
    #         + (1 / self.current_round) * correlation
    #
    #         # Non-linear mapping to node contribution
    #         contribs[i] = self.alpha * (1 - np.exp(-np.exp(-self.alpha
    #                       * (self.local_correlations[client_id] - 1))))
    #
    #     return correlations
    #
    # @staticmethod
    # def process_grad(grads):
    #     """Convert gradients to a flattened 1-D array."""
    #     grads = list(dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())
    #
    #     flattened = grads[0]
    #     for i in range(1, len(grads)):
    #         flattened = np.append(flattened, grads[i])
    #
    #     return flattened
