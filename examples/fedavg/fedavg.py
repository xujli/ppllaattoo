"""
A federated learning training session using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/abs/1812.07108
"""
import os

os.environ['config_file'] = './fedavg_MNIST_mlp.yml'

from plato.servers import fedavg


def main():
    """ A Plato federated learning training session using the FedAtt algorithm. """
    server = fedavg.Server()
    server.run()


if __name__ == "__main__":
    main()
