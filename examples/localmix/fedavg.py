import os

os.environ['config_file'] = './fedavg_MNIST_lenet.yml'

from plato.servers import fedavg
from localmix_client import Client
from localmix_trainer import Trainer

def main():
    """ A Plato federated learning training session using the FedAtt algorithm. """
    server = fedavg.Server()
    client = Client()
    server.run(client)


if __name__ == "__main__":
    main()
