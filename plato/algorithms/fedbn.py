"""
The federated averaging algorithm for PyTorch.
"""
from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """Extract weights from the model."""
        return self.model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        for name in weights.keys():
            if 'bn' not in name:
                self.model.state_dict()[name].data.copy_(weights[name])
