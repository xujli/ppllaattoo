"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config

from plato.servers import fedavg
from plato.algorithms import registry as algorithms_registry
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor
from fedavg_trainer import Trainer
from scipy.stats import wasserstein_distance
from plato.servers.fedavg import *

import numpy as np
import random
import torch


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.lr = Config().trainer.learning_rate
        self.client_momentum = Config().trainer.momentum
        self.batch_nums = Config().data.partition_size // Config().trainer.batch_size
        if Config().data.partition_size % Config().trainer.batch_size != 0:
            self.batch_nums += 1
        self.batch_nums *= Config().trainer.epochs
        self.weight_divergence = {}
        self.occurrence = {}
        self.EMD_distance = {}

    def get_trainer(self, model=None):
        return Trainer(model)

    def init_trainer(self):
        """Setting up the global model to be trained via federated learning."""

        get_trainer = getattr(self, 'get_trainer', trainers_registry.get)
        if self.trainer is None:
            self.trainer = get_trainer(self.model)

        self.trainer.set_client_id(client_id=0)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)

        for i in range(1, Config().clients.total_clients + 1):
            self.weight_divergence[i] = 0
            self.occurrence[i] = 0

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        logging.info("[Server #%d] Configuring the server...", os.getpid())
        super().configure()

        total_rounds = Config().trainer.rounds
        target_accuracy = None
        target_perplexity = None

        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

        if target_accuracy:
            logging.info(
                "Training: %s rounds or accuracy above %.1f%%\n",
                total_rounds,
                100 * target_accuracy,
            )
        elif target_perplexity:

            logging.info(
                "Training: %s rounds or perplexity below %.1f\n",
                total_rounds,
                target_perplexity,
            )
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        self.init_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer
        )

        if not (hasattr(Config().server, "do_test") and not Config().server.do_test):
            if self.datasource is None and self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
            elif self.datasource is None and self.custom_datasource is not None:
                self.datasource = self.custom_datasource()

            self.testset = self.datasource.get_test_set()

            if hasattr(Config().data, "testset_size"):
                # Set the sampler for testset
                import torch

                if hasattr(Config().server, "random_seed"):
                    random_seed = Config().server.random_seed
                else:
                    random_seed = 1

                gen = torch.Generator()
                gen.manual_seed(random_seed)

                all_inclusive = range(len(self.datasource.get_test_set()))
                test_samples = random.sample(all_inclusive, Config().data.testset_size)
                self.testset_sampler = torch.utils.data.SubsetRandomSampler(
                    test_samples, generator=gen
                )
        temp_items = []
        for i in range(1, self.total_clients+1):
            temp_items += ['Client {} EMD'.format(i), 'client {} accuracy'.format(i)]
        # Initialize the csv file which will record results
        result_csv_file = f"{Config().result_dir}" + 'result_{}.csv'.format(Config().data.random_seed)
        csv_processor.initialize_csv(
            result_csv_file, self.recorded_items + temp_items, Config().result_dir
        )

    async def select_clients(self):
        await super().select_clients()
        for selected_client_id in self.selected_clients:
            self.occurrence[selected_client_id] += 1

    def compute_weight_deltas(self, updates):
        weights_received = [payload for (__, __, payload, __) in updates]
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])
        # Get adaptive weighting based on both node contribution and date size
        update_received = self.algorithm.compute_weight_deltas(weights_received)

        select_ids = [id for (id, __, __, __) in updates]
        flattens = []
        for id, update in zip(select_ids, update_received):
            flattened = []
            for params in update.values():
                flattened.extend(torch.flatten(params).numpy())
            flattens.append(flattened)

        flatten_update = np.mean(flattens, 0)
            # self.EMD_distance[id] = self.EMD_distance[id] * (self.occurrence[id] - 1) / self.occurrence[
            #     id] + wasserstein_distance(flattened, flatten_update) / self.occurrence[id]
        for id, flattened in zip(select_ids, flattens):
            self.weight_divergence[id] = np.mean(np.abs(np.array(flattened) - np.array(flatten_update)))

        if Config().clients.do_test:
            self.client_acc = {id: report.accuracy for (id, report, __, __) in updates}

        return update_received

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        # Record results into a .csv file
        new_row = []
        for item in self.recorded_items:
            item_value = self.get_record_items_values()[item]
            new_row.append(item_value)
        if Config().clients.do_test:
            for id in self.weight_divergence.keys():
                new_row.extend([self.weight_divergence[id], self.client_acc[id]])

        result_csv_file = Config().result_dir + 'result_{}.csv'.format(Config().data.random_seed)
        csv_processor.write_csv(result_csv_file, new_row)

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.current_round, self.EMD_distance]
