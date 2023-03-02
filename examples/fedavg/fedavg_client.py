"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import torch
from fedavg_trainer import Trainer
from plato.clients import simple
from copy import deepcopy
import numpy as np
from plato.utils import csv_processor
from scipy.stats import wasserstein_distance
from plato.clients.simple import *


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
        self.flag = np.random.randn(1)
        self.EMD = 0

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

        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer
        )
        result_csv_file = Config().result_dir + 'result_{}_client{}.csv'.format(Config().data.random_seed,
                                                                                self.client_id)

        csv_processor.initialize_csv(
            result_csv_file, ['round', 'accuracy', 'EMD'], Config().result_dir
        )

    def load_payload(self, server_payload):
        " Load model weights and server update direction from server payload onto this client. "
        self.trainer.current_round = server_payload[1]

    def load_payload(self, server_payload):
        " Load model weights and server update direction from server payload onto this client. "
        self.trainer.gap = server_payload[1] - self.trainer.current_round
        print(self.client_id, server_payload[1], self.trainer.current_round)
        self.trainer.current_round = server_payload[1]
        self.current_round = server_payload[1]
        self.trainer.last_model = deepcopy(self.algorithm.extract_weights())
        self.algorithm.load_weights(server_payload[0])
        self.weight = server_payload[0]

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info(
            fonts.colourize(
                f"[{self}] Started training in communication round #{self.current_round}."
            )
        )

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError as exc:
            logging.info(
                fonts.colourize(f"[{self}] Error occurred during training: {exc}")
            )
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if (hasattr(Config().clients, "do_test") and Config().clients.do_test) and (
                not hasattr(Config().clients, "test_interval")
                or self.current_round % Config().clients.test_interval == 0
        ):

            flatten_global = []
            for name, params in self.weight.items():
                flatten_global.extend(torch.flatten(params))

            flatten_client = []
            for name, params in weights.items():
                flatten_client.extend(torch.flatten(params))

            self.EMD += wasserstein_distance(flatten_client, flatten_global)

            accuracy = self.trainer.test(self.testset, self.testset_sampler)
            item_value = {
                'round':
                    self.current_round,
                'accuracy':
                    accuracy * 100,
                'EMD':
                    self.EMD / self.current_round
            }

            result_csv_file = Config().result_dir + 'result_{}_client{}.csv'.format(Config().data.random_seed,
                                                                                    self.client_id)

            csv_processor.write_csv(result_csv_file, list(item_value.values()))
            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            if hasattr(Config().trainer, "target_perplexity"):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self, 100 * accuracy)
        else:
            accuracy = 0

        comm_time = time.time()

        if (
                hasattr(Config().clients, "sleep_simulation")
                and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time
            self.report = Report(
                self.sampler.trainset_size(),
                accuracy,
                (avg_training_time + sleep_seconds) * Config().trainer.epochs,
                comm_time,
                False,
            )
        else:
            self.report = Report(
                self.sampler.trainset_size(), accuracy, training_time, comm_time, False
            )

        return self.report, weights
