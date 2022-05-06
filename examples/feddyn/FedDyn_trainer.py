"""
The training and testing loops for PyTorch.
"""
import asyncio
import logging
import multiprocessing as mp
import os
import copy
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

        self.local_delta = {}


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
                    train_loader = torch.utils.data.DataLoader(
                        dataset=trainset,
                        shuffle=False,
                        batch_size=batch_size,
                        sampler=sampler.get())

                iterations_per_epoch = np.ceil(len(trainset) /
                                               batch_size).astype(int)
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()

                fixed_model = copy.deepcopy(self.model)
                for param_t in fixed_model.parameters():
                    param_t.requires_grad = False
                fixed_params = {n: p for n, p in fixed_model.named_parameters()}

                if not bool(self.local_delta):
                    for n, p in fixed_params.items():
                        self.local_delta[n] = torch.zeros(p.shape)
                # Initializing the loss criterion
                _loss_criterion = getattr(self, "loss_criterion", None)
                if callable(_loss_criterion):
                    loss_criterion = self.loss_criterion(self.model)
                else:
                    loss_criterion = nn.CrossEntropyLoss()

                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        self.optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None

                for epoch in range(1, epochs + 1):
                    for batch_id, (examples,
                                   labels) in enumerate(train_loader):
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        self.optimizer.zero_grad()

                        if cut_layer is None:
                            outputs = self.model(examples)
                        else:
                            outputs = self.model.forward_from(
                                examples, cut_layer)

                        loss = loss_criterion(outputs, labels)

                        ## Weight L2 loss
                        reg_loss = 0
                        for n, p in self.model.named_parameters():
                            reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()

                        ## local gradient regularization
                        lg_loss = 0
                        for n, p in self.model.named_parameters():
                            p = torch.flatten(p)
                            local_d = self.local_delta[n].detach().clone().to(self.device)
                            local_grad = torch.flatten(local_d)
                            lg_loss += (p * local_grad.detach()).sum()

                        loss = loss - lg_loss + 0.5 * Config().trainer.mu * reg_loss
                        loss.backward()

                        self.optimizer.step()

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

                    if hasattr(self.optimizer, "params_state_update"):
                        self.optimizer.params_state_update()

                # Update Local Delta
                for n, p in self.model.named_parameters():
                    self.local_delta[n] = (
                                self.local_delta[n] - Config().trainer.mu * (p - fixed_params[n]).detach().clone().to('cpu'))

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            print(training_exception)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()
