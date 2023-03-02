"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from fedtrip_trainer import Trainer
from plato.clients import simple
from plato.algorithms import registry as algorithms_registry
from plato.processors import registry as processor_registry
from plato.trainers import registry as trainers_registry
from copy import deepcopy
from plato.config import Config
import numpy as np

class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.interval = None
        self.server_update_direction = []
        self.flag = np.random.randn(1)

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

        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer
        )

    def load_payload(self, server_payload):
        " Load model weights and server update direction from server payload onto this client. "
        self.trainer.gap = server_payload[1] - self.trainer.current_round
        self.trainer.current_round = server_payload[1]
        self.trainer.last_model = deepcopy(self.algorithm.extract_weights())
        self.algorithm.load_weights(server_payload[0])
        EMDs = server_payload[2]
        EMD_values = list(EMDs.values())

        if (self.trainer.EMD != 0) and (float(self.current_round) / Config().trainer.rounds > 0.1):
            self.trainer.EMD = EMDs[self.client_id]
            self.trainer.mean = np.sum(EMD_values) / np.sum(np.array(EMD_values) != 0)
            self.trainer.coef = self.trainer.EMD / self.trainer.mean
            print(self.client_id, self.trainer.coef)


    async def train(self):
        report, weights = await super().train()
        return report, [weights, self.client_id]
