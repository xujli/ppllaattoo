"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import logging
import time
from plato.config import Config
from localmix_trainer import Trainer
from plato.clients import simple
from plato.algorithms import registry as algorithms_registry
from plato.datasources import registry as datasources_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry
from torchtext.data.functional import to_map_style_dataset
from copy import deepcopy


class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.interval = None
        self.server_update_direction = []

    def get_trainer(self, model=None):
        return Trainer(model)

    def configure(self) -> None:
        """Prepare this client for training."""
        get_trainer = getattr(self, 'get_trainer', trainers_registry.get)
        if self.trainer is None:
            self.trainer = get_trainer(self.model)
        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)
        self.algorithm.set_client_id(self.client_id)


    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        data_loading_start_time = time.perf_counter()
        logging.info("[Client #%d] Loading its data source...", self.client_id)

        if self.datasource is None:
            self.datasource = datasources_registry.get(client_id=self.client_id)

        self.data_loaded = True

        logging.info("[Client #%d] Dataset size: %s", self.client_id,
                     self.datasource.num_train_examples())

        # Setting up the data sampler
        self.sampler = samplers_registry.get(self.datasource, self.client_id)
        samlpers = Config().clients.total_clients
        self.samplers = {}
        for id in range(1, samlpers+1):
            self.samplers[id] = samplers_registry.get(self.datasource, id)

        if hasattr(Config().trainer, 'use_mindspore'):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()

        self.data_loading_time = time.perf_counter() - data_loading_start_time

        if Config().data.datasource == 'IMDB':
            self.trainset = to_map_style_dataset(self.trainset)
            self.trainer.configure_IMDB(self.trainset)

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info("[Client #%d] Started training.", self.client_id)
        self.trainer.samplers = self.samplers
        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError:
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset)

            if accuracy == 0:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            logging.info("[Client #{:d}] Test accuracy: {:.2f}%".format(
                self.client_id, 100 * accuracy))
        else:
            accuracy = 0

        data_loading_time = 0

        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True

        return simple.Report(self.sampler.trainset_size(), accuracy, training_time,
                      data_loading_time), weights
