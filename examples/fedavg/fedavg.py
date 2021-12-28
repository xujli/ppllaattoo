import os

os.environ['config_file'] = './fedavg_FashionMNIST_lenet5.yml'

from plato.servers import fedavg


def main():
    """ A Plato federated learning training session using the FedAtt algorithm. """
    server = fedavg.Server()
    server.run()


if __name__ == "__main__":
    main()
