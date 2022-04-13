"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import os
import torch
import logging
import numpy as np
import torch.nn as nn
from plato.config import Config
from plato.trainers import basic

from plato.utils import optimizers


class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client. """
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model)
        self.server_update_direction = []

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
                    train_loader = self.train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:
                    if config['datasource'] == 'IMDB':
                        train_loader = torch.utils.data.DataLoader(
                            dataset=trainset,
                            shuffle=False,
                            batch_size=batch_size,
                            sampler=sampler.get(),
                            collate_fn=self.collate_batch,
                        )
                    else:
                        train_loader = torch.utils.data.DataLoader(
                            dataset=trainset,
                            shuffle=False,
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

                # Initializing the optimizer
                get_optimizer = getattr(self, "get_optimizer",
                                        optimizers.get_optimizer)
                optimizer = get_optimizer(self.model)

                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None
                all_labels = []

                if self.server_update_direction is None:
                    self.server_update_direction = {}
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            self.server_update_direction[p] = torch.zeros(p.shape).to(self.device)

                for epoch in range(1, epochs + 1):
                    for batch_id, item in enumerate(train_loader):
                        if config['datasource'] == 'IMDB':
                            examples, labels, offsets = item
                            examples, labels, offsets = examples.to(self.device), labels.to(
                                self.device), offsets.to(self.device)
                        else:
                            examples, labels = item
                            examples, labels = examples.to(self.device), labels.to(
                                self.device)

                        optimizer.zero_grad()

                        for group in optimizer.param_groups:
                            for p, update in zip(group['params'], self.server_update_direction.values()):
                                optimizer.state[p]['momentum_buffer'] = update.to(self.device)
                        all_labels.extend(labels.cpu().numpy())

                        if cut_layer is None:
                            outputs = self.model(examples) if config['datasource'] != 'IMDB' else \
                                self.model(examples, offsets)
                        else:
                            outputs = self.model.forward_from(
                                examples, cut_layer)

                        loss = loss_criterion(outputs, labels)

                        loss.backward()

                        optimizer.step()

                        if lr_schedule is not None:
                            lr_schedule.step()

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss.data.item()})

                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))

                    if hasattr(optimizer, "params_state_update"):
                        optimizer.params_state_update()

        except Exception as training_exception:
            logging.info(training_exception)
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()
