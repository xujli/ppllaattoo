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
import torch.nn.functional as F

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers
from copy import deepcopy

class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client. """
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model)
        self.update_direction = None
        self.norms = []
        self.selected = False
        self.diff = None
        self.momentum = Config().trainer.momentum
        self.current_round = 0
        self.gap = 0
        self.samplers = {}

    def get_image_means(self, trainset, batch_size=5):
        global_images, global_labels = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
        for id, sampler in self.samplers.items():
            if id != self.client_id:
                train_dloader = torch.utils.data.DataLoader(
                    dataset=trainset,
                    shuffle=False,
                    batch_size=batch_size,
                    sampler=sampler.get()
                )

                images_means, labels_means = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
                for batch_idx, (images, labels) in enumerate(train_dloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    images_mean = torch.mean(images, dim=0).unsqueeze(0)
                    labels_mean = torch.mean(F.one_hot(labels, num_classes=10).float(), dim=0).unsqueeze(0)
                    images_means = torch.cat([images_means, images_mean], dim=0)
                    labels_means = torch.cat([labels_means, labels_mean], dim=0)
                global_images = torch.cat([global_images, images_means], dim=0)
                global_labels = torch.cat([global_labels, labels_means], dim=0)

        return global_images, global_labels

    def mix_data(self, x, y, alpha=0.1):
        # if alpha > 0:
        #     lam = np.random.beta(alpha, alpha)
        # else:
        #     lam = 1
        lam = 1 - alpha
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        target = y[index]
        return mixed_x, target, lam

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
        self.pure_gradients = None
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
                    train_dloader = _train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:
                    train_dloader = torch.utils.data.DataLoader(
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
                        optimizer, iterations_per_epoch, train_dloader)
                else:
                    lr_schedule = None
                all_labels = []
                #

                self.global_model = deepcopy(self.model.state_dict())
                for epoch in range(1, epochs + 1):
                    for batch_id, item in enumerate(train_dloader):
                        examples, labels = item
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)

                        mix_examples, perm_labels, lam = self.mix_data(examples, labels)

                        optimizer.zero_grad()

                        all_labels.extend(labels.cpu().numpy())

                        if cut_layer is None:
                            outputs = self.model(mix_examples)
                        else:
                            outputs = self.model.forward_from(
                                mix_examples, cut_layer)

                        loss = lam * loss_criterion(outputs, labels) + (1 - lam) * loss_criterion(outputs, perm_labels)

                        loss.backward()

                        optimizer.step()

                        if lr_schedule is not None:
                            lr_schedule.step()

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_dloader),
                                            loss.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss.data.item()})

                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_dloader),
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
