"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import os

os.environ['config_file'] = 'fedtrip_MNIST_mlp.yml'

import fedtrip_client
import fedtrip_server

import torch
import numpy as np
import random
from plato.config import Config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
setup_seed(Config().data.random_seed)

def main():
    client = fedtrip_client.Client()
    server = fedtrip_server.Server()

    server.run(client)


if __name__ == "__main__":
    main()