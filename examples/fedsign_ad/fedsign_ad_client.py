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
        self.interval = 1
        self.server_update_direction = []

    def load_payload(self, server_payload):
        " Load model weights and server update direction from server payload onto this client. "
        old_weights = self.algorithm.extract_weights()
        self.server_update_direction = []
        self.interval = server_payload[1][self.client_id]
        for (name, old_weight), (name, new_weight) in zip(old_weights.items(), server_payload[0].items()):
            if ('running_mean' not in name) and ('running_var' not in name) and ('num_batches_tracked' not in name):
                self.server_update_direction.append((new_weight - old_weight) / self.interval)
        # print(len(self.server_update_direction))
        self.trainer.server_update_direction = self.server_update_direction
        self.algorithm.load_weights(server_payload[0])