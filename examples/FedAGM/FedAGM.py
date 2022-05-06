"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import os
import torch
import random
import numpy as np
from plato.config import Config
os.environ['config_file'] = 'FedAGM_EMNIST_lenet5.yml'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
setup_seed(Config().data.random_seed)

import FedAGM_client
import FedAGM_server


def main():
    """ A Plato federated learning training session using the SCAFFOLD algorithm. """
    client = FedAGM_client.Client()
    server = FedAGM_server.Server()

    server.run(client)


if __name__ == "__main__":
    main()
