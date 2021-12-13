"""The LeNet-5 model for PyTorch.

Reference:

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to
document recognition." Proceedings of the IEEE, November 1998.
"""
import collections

import torch.nn as nn
import torch.nn.functional as F

from plato.config import Config


class Model(nn.Module):
    """The LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCun paper
        self.fc1 = nn.Linear(784, num_classes)
        # self.bn = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, num_classes)

    def flatten(self, x):
        """Flatten the tensor."""
        return x.view(x.size(0), -1)

    def forward(self, x):
        """Forward pass."""
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.bn(x)
        # x = self.fc2(x)

        return F.log_softmax(x, dim=1)


    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        if hasattr(Config().trainer, 'num_classes'):
            return Model(num_classes=Config().trainer.num_classes)
        return Model()
