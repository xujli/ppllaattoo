"""
A federated learning training session using the FedAdp algorithm.

Reference:

Wu et al., "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Transactions on Cognitive Communications and Networking (TCCN'21).

https://ieeexplore.ieee.org/abstract/document/9442814
"""
import os
import torch
import random
import numpy as np
from plato.config import Config
os.environ['config_file'] = './fedprox_MNIST_lenet5.yml'

import fedadp_server

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(Config().data.random_seed)

def main():
    """ A Plato federated learning training session using the FedAdp algorithm. """
    server = fedadp_server.Server()
    server.run()


if __name__ == "__main__":
    main()
