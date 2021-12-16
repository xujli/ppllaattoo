"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.interval_dict = {
            client + 1: 1
            for client in range(self.total_clients)
        }

    def extract_client_updates(self, updates):
        """Extract the model weight updates from client updates."""
        weights_received = [payload for (__, payload) in updates]
        for client in self.clients.keys():
            if client in self.selected_clients:
                self.interval_dict[client] = 1
            else:
                self.interval_dict[client] += 1
        return self.algorithm.compute_weight_updates(weights_received)

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.interval_dict]


