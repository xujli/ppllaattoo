"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import os

os.environ['config_file'] = 'fedsign_ad_MNIST_lenet5.yml'

import fedsign_ad_client
import fedsign_ad_server


def main():
    """ A Plato federated learning training session using the SCAFFOLD algorithm. """
    client = fedsign_ad_client.Client()
    server = fedsign_ad_server.Server()

    server.run(client)


if __name__ == "__main__":
    main()