"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

from FedCM_trainer import Trainer
from plato.algorithms import registry as algorithms_registry
from plato.trainers import registry as trainers_registry
from plato.clients import simple


class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)


    def get_trainer(self, model=None):
        return Trainer(model)

    def configure(self) -> None:
        """Prepare this client for training."""
        get_trainer = getattr(self, 'get_trainer', trainers_registry.get)
        if self.trainer is None:
            self.trainer = get_trainer(self.model)
        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)
        self.algorithm.set_client_id(self.client_id)

    def load_payload(self, server_payload):
        " Load model weights and server update direction from server payload onto this client. "
        self.server_update_direction = server_payload[1]
        self.trainer.server_update_direction = self.server_update_direction
        self.algorithm.load_weights(server_payload[0])
