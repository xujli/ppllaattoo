"""
A federated learning training session using the FedAdp algorithm.

Reference:

Wu et al., "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Transactions on Cognitive Communications and Networking (TCCN'21).

https://ieeexplore.ieee.org/abstract/document/9442814
"""
import numpy as np
from scipy.stats import wasserstein_distance
from plato.servers import fedavg

class Server(fedavg.Server):
    """A federated learning server using the FedAdp algorithm."""
    def __init__(self):
        super().__init__()

        # alpha controls the decreasing rate of the mapping function
        self.alpha = 5
        self.local_correlations = {}
        self.last_global_grads = None
        self.adaptive_weighting = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload for (__, payload) in updates]

        num_samples = [report.num_samples for (report, __) in updates]
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])
        # Get adaptive weighting based on both node contribution and date size
        updates_received = self.algorithm.compute_weight_updates(weights_received)
        self.adaptive_weighting = self.calc_adaptive_weighting(updates_received, num_samples)

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """
        # Extract weights udpates from the client updates
        weights_received = self.extract_client_updates(updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # Use adaptive weighted average
        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                avg_update[name] += delta * self.adaptive_weighting[i]

        return avg_update

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