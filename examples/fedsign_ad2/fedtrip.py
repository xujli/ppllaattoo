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


def main():
    client = fedtrip_client.Client()
    server = fedtrip_server.Server()

    server.run(client)


if __name__ == "__main__":
    main()