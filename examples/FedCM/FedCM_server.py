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
from FedCM_trainer import Trainer


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.beta = Config().trainer.beta
        self.lr = Config().trainer.learning_rate
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        self.momentum_update_direction = {}

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


    async def federated_averaging(self, updates):
        # update (report, (weights, ...))
        """ Aggregate weight updates and deltas updates from the clients. """

        # Perform weighted averaging
        avg_updates = await super(Server, self).federated_averaging(updates)

        if self.momentum_update_direction is None:
            self.momentum_update_direction = {}

        # Use adaptive weighted average
        for name, delta in avg_updates.items():
            if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                continue
            self.momentum_update_direction[name] = (-delta) / (self.lr * self.batch_nums)

        for name, delta in avg_updates.items():
            if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                continue
            avg_updates[name] = - self.beta * self.batch_nums * self.momentum_update_direction[name]

        return avg_updates

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.momentum_update_direction]