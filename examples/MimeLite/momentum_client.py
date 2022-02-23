"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import logging
from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client co   ntrol variate."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        self.client_update_direction = None

    async def train(self):
        report, weights = await super().train()
        return report, [weights, self.trainer.momentums]

    def load_payload(self, server_payload):
        " Load model weights and server update direction from server payload onto this client. "
        self.algorithm.load_weights(server_payload[0])
        self.trainer.update_direction = server_payload[1]

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info("[Client #%d] Started training.", self.client_id)

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
                      data_loading_time), [weights, self.trainer.full_batch_grad]
