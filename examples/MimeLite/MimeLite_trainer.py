"""
The training and testing loops for PyTorch.
"""
import asyncio
import logging
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """A basic federated learning trainer, used by both the client and the server."""
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__()

        if model is None:
            model = models_registry.get()

        # Use data parallelism if multiple GPUs are available and the configuration specifies it
        if Config().is_parallel():
            logging.info("Using Data Parallelism.")
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model)
        else:
            self.model = model

        # Initializing the optimizer, this optimizer will used in the whole training loop
        get_optimizer = getattr(self, "get_optimizer",
                                optimizers.get_optimizer)
        self.optimizer = get_optimizer(self.model)
        self.full_batch_grad = []
        self.update_direction = None
        self.train_dloader = None
        self.test_dloader = None

    def set_momentum_params(self):
        if not self.update_direction is None:
            for group, momentum in zip(self.optimizer.param_groups, self.update_direction):
                # through the models trained by the opt
                for idx, p in enumerate(group['params']):
                    param_state = self.optimizer.state[p]
                    param_state['momentum_buffer'] = torch.tensor(momentum[idx].clone().detach()).to(self.device)

    def get_momentum_params(self):
        self.momentums = []

        for group in self.optimizer.param_groups:
            momentum = []
            for p in group['params']:
                param_state = self.optimizer.state[p]
                momentum.append(param_state['momentum_buffer'].detach().clone().cpu().numpy())

            self.momentums.append(momentum)

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        if 'use_wandb' in config:
            import wandb

            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        try:
            custom_train = getattr(self, "train_model", None)

            if callable(custom_train):
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                log_interval = 10
                batch_size = config['batch_size']

                logging.info("[Client #%d] Loading the dataset.",
                             self.client_id)
                _train_loader = getattr(self, "train_loader", None)


                if callable(_train_loader):
                    self.train_dloader = self.train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:

                    self.train_dloader = torch.utils.data.DataLoader(
                        dataset=trainset,
                        shuffle=False,
                        num_workers=4,
                        batch_size=batch_size,
                        sampler=sampler.get()
                    )

                iterations_per_epoch = np.ceil(len(trainset) /
                                               batch_size).astype(int)
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()

                # Initializing the loss criterion
                _loss_criterion = getattr(self, "loss_criterion", None)
                if callable(_loss_criterion):
                    loss_criterion = self.loss_criterion(self.model)
                else:
                    loss_criterion = nn.CrossEntropyLoss()

                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        self.optimizer, iterations_per_epoch, self.train_dloader)
                else:
                    lr_schedule = None

                for epoch in range(1, epochs + 1):
                    for batch_id, (examples,
                                   labels) in enumerate(self.train_dloader):
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        self.optimizer.zero_grad()
                        self.set_momentum_params()

                        if cut_layer is None:
                            outputs = self.model(examples)
                        else:
                            outputs = self.model.forward_from(
                                examples, cut_layer)

                        loss = loss_criterion(outputs, labels)

                        loss.backward()

                        self.optimizer.step()

                        if lr_schedule is not None:
                            lr_schedule.step()

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(self.train_dloader),
                                            loss.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss.data.item()})

                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(self.train_dloader),
                                            loss.data.item()))

                    if hasattr(self.optimizer, "params_state_update"):
                        self.optimizer.params_state_update()

                for batch_id, (examples,
                               labels) in enumerate(self.train_dloader):
                    examples, labels = examples.to(self.device), labels.to(
                        self.device)
                    self.optimizer.zero_grad()

                    if cut_layer is None:
                        outputs = self.model(examples)
                    else:
                        outputs = self.model.forward_from(
                            examples, cut_layer)

                    loss = loss_criterion(outputs, labels) / len(self.train_dloader)

                    loss.backward()

                self.full_batch_grad = [param.grad.data for name, param in self.model.named_parameters()]

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()



    def train(self, trainset, sampler, cut_layer=None) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.

        Returns:
        float: Elapsed time during training.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        if hasattr(Config().trainer, 'max_concurrency'):
            self.start_training()
            tic = time.perf_counter()

            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            train_proc = mp.Process(target=self.train_process,
                                    args=(
                                        config,
                                        trainset,
                                        sampler,
                                        cut_layer,
                                    ))
            train_proc.start()
            train_proc.join()


            model_name = Config().trainer.model_name
            filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"

            try:
                self.load_model(filename)
            except OSError as error:  # the model file is not found, training failed
                if hasattr(Config().trainer, 'max_concurrency'):
                    self.run_sql_statement(
                        "DELETE FROM trainers WHERE run_id = (?)",
                        (self.client_id, ))
                raise ValueError(
                    f"Training on client {self.client_id} failed.") from error

            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler, cut_layer)
            toc = time.perf_counter()

        self.get_momentum_params()
        training_time = toc - tic
        return training_time

    def test_process(self, config, testset):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        """
        self.model.to(self.device)
        self.model.eval()


        try:
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                self.test_dloader = torch.utils.data.DataLoader(
                    testset, batch_size=config['batch_size'], num_workers=4, shuffle=False)

                correct = 0
                total = 0

                with torch.no_grad():
                    for examples, labels in self.test_dloader:
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)

                        outputs = self.model(examples)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if 'max_concurrency' in config:
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def test(self, testset) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        if hasattr(Config().trainer, 'max_concurrency'):
            self.start_training()

            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            proc = mp.Process(target=self.test_process,
                              args=(
                                  config,
                                  testset,
                              ))
            proc.start()
            proc.join()

            try:
                model_name = Config().trainer.model_name
                filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
                accuracy = self.load_accuracy(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Testing on client #{self.client_id} failed.") from error

            self.pause_training()
        else:
            accuracy = self.test_process(config, testset)

        return accuracy

    # async def server_test(self, testset):
    #     """Testing the model on the server using the provided test dataset.
    # 
    #     Arguments:
    #     testset: The test dataset.
    #     """
    #     config = Config().trainer._asdict()
    #     config['run_id'] = Config().params['run_id']
    # 
    #     self.model.to(self.device)
    #     self.model.eval()
    # 
    #     custom_test = getattr(self, "test_model", None)
    # 
    #     if callable(custom_test):
    #         return self.test_model(config, testset)
    # 
    #     test_loader = torch.utils.data.DataLoader(
    #         testset, batch_size=config['batch_size'], shuffle=False)
    # 
    #     correct = 0
    #     total = 0
    # 
    #     with torch.no_grad():
    #         for examples, labels in test_loader:
    #             examples, labels = examples.to(self.device), labels.to(
    #                 self.device)
    # 
    #             outputs = self.model(examples)
    # 
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    # 
    #             # Yield to other tasks in the server
    #             await asyncio.sleep(0)
    # 
    #     return correct / total
