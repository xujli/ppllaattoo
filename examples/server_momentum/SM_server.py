"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config
from plato.servers import fedavg

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.momentum_update_direction = None
        self.beta = Config().trainer.beta
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        if Config().data.partition_size % Config().trainer.batch_size != 0:
            self.batch_nums += 1

    async def federated_averaging(self, updates):
        # update (report, (weights, ...))
        """ Aggregate weight updates and deltas updates from the clients. """

        # Perform weighted averaging
        avg_updates = await super(Server, self).federated_averaging(updates)

        if self.momentum_update_direction is None:
            self.momentum_update_direction = {}
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                self.momentum_update_direction[name] = delta / self.batch_nums
        else:
            # Use adaptive weighted average
            for name, delta in avg_updates.items():
                self.momentum_update_direction[name] = (1 - self.beta) * delta / self.batch_nums + \
                    self.beta * self.momentum_update_direction[name]
                avg_updates[name] = self.momentum_update_direction[name]

        return avg_updates

