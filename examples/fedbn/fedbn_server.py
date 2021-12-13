"""
A federated learning server using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/abs/1812.07108
"""
from collections import OrderedDict

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import norm
import torch
from plato.servers import fedavg


class Server(fedavg.Server):
    """ A federated learning server using the FedAtt algorithm. """
    def __init__(self):
        super().__init__()

        # epsilon: step size for aggregation
        self.epsilon = 1e-5

        # dp: the magnitude of normal noise in the randomization mechanism
        self.dp = 0.001

    async def federated_averaging(self, updates):
        """ Aggregate weight updates from the clients using FedAtt. """
        # Extract weights from the updates
        weights_received = [payload for (__, payload) in updates]

        # Extract baseline model weights
        baseline_weights = self.algorithm.extract_weights()

        num_samples = [report.num_samples for (report, __) in updates]

        # Update server weights
        update = self.avg_att(baseline_weights, weights_received, num_samples)

        return update

    def avg_att(self, baseline_weights, weights_received, num_samples):
        """ Perform attentive aggregation with the attention mechanism. """
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in baseline_weights.items()
        }
        similarity = {}

        for i, update in enumerate(weights_received):
            for name, weights in update.items():
                avg_update[name] += avg_update[name] * num_samples[i] / np.sum(num_samples)

        for name, weights in avg_update.items():
            if ('bn' in name) and (name.split('.')[0] not in similarity.keys()):
                bn_name = name.split('.')[0]




