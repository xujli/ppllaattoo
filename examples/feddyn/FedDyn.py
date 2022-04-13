"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import os
os.environ['config_file'] = 'FedDyn_MNIST_lenet5.yml'


import FedDyn_client
import FedDyn_server


def main():
    """ A Plato federated learning training session using the SCAFFOLD algorithm. """
    client = FedDyn_client.Client()
    server = FedDyn_server.Server()

    server.run(client)


if __name__ == "__main__":
    main()
