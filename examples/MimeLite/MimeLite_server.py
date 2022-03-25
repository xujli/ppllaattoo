"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

from plato.servers import fedavg
from plato.config import Config
from scipy.stats import wasserstein_distance
import numpy as np


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.momentum_update_direction = None

        # alpha controls the decreasing rate of the mapping function
        self.alpha = 2
        self.local_correlations = {}
        self.last_global_grads = None
        self.adaptive_weighting = None
        self.momentum = Config().trainer.momentum

    def extract_client_updates(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload[0] for (__, payload) in updates]

        self.momentum_direction_received = [
            payload[1] for (__, payload) in updates
        ]
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])
        # Get adaptive weighting based on both node contribution and date size
        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        # update (report, (weights, ...))
        """ Aggregate weight updates and deltas updates from the clients. """
        weights_received = self.extract_client_updates(updates)
        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        num_samples = [report.num_samples for (report, __) in updates]
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])
        # Use adaptive weighted average
        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                avg_update[name] += delta * num_samples[i] / self.total_samples

        avg_gradient_direction = []
        avg_updates = []
        for update in self.momentum_direction_received[0]:
            avg_updates.append(self.trainer.zeros(update.shape))

        avg_gradient_direction.append(avg_updates)

        for i, update in enumerate(self.momentum_direction_received):
            for idx2, delta in enumerate(update):
                avg_gradient_direction[0][idx2] += delta * num_samples[i] / self.total_samples

        if self.momentum_update_direction is None:
            self.momentum_update_direction = avg_gradient_direction
        else:
            for i, update in enumerate(avg_gradient_direction):
                for idx2, delta in enumerate(update):
                    self.momentum_update_direction[0][idx2] = self.momentum * self.momentum_update_direction[0][idx2] + \
                        avg_gradient_direction[0][idx2]

        return avg_update

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.momentum_update_direction]

    def calc_adaptive_weighting(self, updates, num_samples):
        """ Compute the weights for model aggregation considering both node contribution
        and data size. """
        # Get the node contribution
        contribs = self.calc_contribution(updates, num_samples)

        # Calculate the weighting of each participating client for aggregation
        adaptive_weighting = np.exp(contribs) / np.sum(np.exp(contribs))

        return adaptive_weighting

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
            correlations[i] = - wasserstein_distance(local_grads, curr_global_grads)

        return correlations

    @staticmethod
    def process_grad(grads):
        """Convert gradients to a flattened 1-D array."""
        grads = list(dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for i in range(1, len(grads)):
            flattened = np.append(flattened, grads[i])

        return flattened

    # def avg_att(self, baseline_weights, weights_received):
    #     """ Perform attentive aggregation with the attention mechanism. """
    #     att_update = {
    #         name: self.trainer.zeros(weights.shape)
    #         for name, weights in weights_received[0].items()
    #     }
    #
    #     atts = OrderedDict()
    #     for name, weight in baseline_weights.items():
    #         atts[name] = self.trainer.zeros(len(weights_received))
    #         for i, update in enumerate(weights_received):
    #             delta = update[name]
    #             atts[name][i] = torch.linalg.norm(delta)
    #
    #     for name in baseline_weights.keys():
    #         atts[name] = F.softmax(atts[name], dim=0)
    #
    #     for name, weight in baseline_weights.items():
    #         att_weight = self.trainer.zeros(weight.shape)
    #         for i, update in enumerate(weights_received):
    #             delta = update[name]
    #             att_weight += torch.mul(delta, atts[name][i])
    #
    #         att_update[name] = torch.mul(att_weight, self.epsilon) + torch.mul(
    #             torch.randn(weight.shape), self.dp)