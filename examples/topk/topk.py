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
os.environ['config_file'] = 'topk_MNIST_lenet5.yml'

import momentum_client
import momentum_server
import momentum_trainer
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(Config().data.random_seed)

def main():
    """ A Plato federated learning training session using the SCAFFOLD algorithm. """
    trainer = momentum_trainer.Trainer()
    client = momentum_client.Client(trainer=trainer)
    server = momentum_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
