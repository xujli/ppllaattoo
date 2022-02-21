"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

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

        self.client_update_direction = None

    async def train(self):
        report, weights = await super().train()
        return report, [weights, self.trainer.momentums]

    def load_payload(self, server_payload):
        " Load model weights and server update direction from server payload onto this client. "
        self.algorithm.load_weights(server_payload[0])
        self.trainer.set_momentum_params(server_payload[1])
