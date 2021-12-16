"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import os

os.environ['config_file'] = 'fedsign_IMDB_RNN.yml'

import fedsign_client
import fedsign_trainer
from fedsign_server import Server

def main():
    """ A Plato federated learning training session using the SCAFFOLD algorithm. """
    trainer = fedsign_trainer.Trainer()
    client = fedsign_client.Client(trainer=trainer)
    server = Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()