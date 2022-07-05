import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format
from plato.config import Config


class MLP(nn.Module):
    """The LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCun paper
        self.fc1 = nn.Linear(784, 100)
        # self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_classes)

    def flatten(self, x):
        """Flatten the tensor."""
        return x.view(x.size(0), -1)

    def forward(self, x):
        """Forward pass."""
        x = self.flatten(x)
        proj = self.fc1(x)
        x = F.relu(proj)
        # x = self.bn(x)
        x = self.fc2(x)

        return x

class CNN(nn.Module):
    """The LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCun paper
        self.conv1 = nn.Conv2d(in_channels=1 if 'MNIST' in Config().data.datasource else 3,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=5,
                               bias=True)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120 if 'MNIST' in Config().data.datasource else 480, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict['conv1'] = self.conv1
        self.layerdict['relu1'] = self.relu1
        self.layerdict['pool1'] = self.pool1
        self.layerdict['conv2'] = self.conv2
        self.layerdict['relu2'] = self.relu2
        self.layerdict['pool2'] = self.pool2
        self.layerdict['conv3'] = self.conv3
        self.layerdict['relu3'] = self.relu3
        self.layerdict['flatten'] = self.flatten
        self.layerdict['fc4'] = self.fc4
        self.layerdict['relu4'] = self.relu4
        self.layerdict['fc5'] = self.fc5
        self.layers.append('conv1')
        self.layers.append('relu1')
        self.layers.append('pool1')
        self.layers.append('conv2')
        self.layers.append('relu2')
        self.layers.append('pool2')
        self.layers.append('conv3')
        self.layers.append('relu3')
        self.layers.append('flatten')
        self.layers.append('fc4')
        self.layers.append('relu4')
        self.layers.append('fc5')

    def flatten(self, x):
        """Flatten the tensor."""
        return x.view(x.size(0), -1)

    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)

        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    inputs = torch.randn(1, 1, 28, 28)
    model = CNN()
    total_ops, total_params = profile(model, inputs=(inputs, ))
    total_ops, total_params = clever_format([total_ops, total_params], '%.3f')
    print(total_ops, total_params)