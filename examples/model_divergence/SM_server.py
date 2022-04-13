"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config
from plato.servers import fedavg
import numpy as np
from scipy.stats import wasserstein_distance

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.momentum_update_direction = None
        self.beta = Config().trainer.beta
        self.lr = Config().trainer.learning_rate
        self.alpha = 1
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        if Config().data.partition_size % Config().trainer.batch_size != 0:
            self.batch_nums += 1

    def extract_client_updates(self, updates):
        """Extract the model weight updates from client updates."""
        weights_received = [payload for (__, payload) in updates]
        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):

        weights_received = self.extract_client_updates(updates)
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])

        for name in weights_received[0].keys():
            divergence = np.zeros((len(weights_received), len(weights_received)))
            for i in range(len(weights_received) - 1):
                for j in range(i + 1, len(weights_received)):
                    divergence[i, j] = np.mean(np.abs(weights_received[i][name].numpy() - weights_received[j][name].numpy()))
                    divergence[j, i] = divergence[i, j]

            print(name, np.sum(divergence))
        avg_updates = await super(Server, self).federated_averaging(updates)

        return avg_updates

