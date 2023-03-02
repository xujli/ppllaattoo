"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import os
import torch
import asyncio
import logging
import numpy as np
import torch.nn as nn

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers
from copy import deepcopy
from openTSNE import TSNE
import matplotlib.pyplot as plt

def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    return ax

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
        self.current_round = 0
        self.gap = 0
        self.dl = None

    def test_process(self, config, testset, sampler=None):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        """
        self.model.to(self.device)
        self.model.eval()
        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = nn.CrossEntropyLoss()
        try:
            custom_test = getattr(self, "test_testmodel", None)


            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                test_loader = torch.utils.data.DataLoader(
                    dataset=testset,
                    shuffle=False,
                    batch_size=config['batch_size'],
                )

                correct = 0
                total = 0

                with torch.no_grad():
                    for item in test_loader:
                        examples, labels = item
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        outputs = self.model(examples)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total

            if self.current_round >= Config().trainer.rounds - 1:
                self.tsne_vis(test_loader=test_loader)

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

    async def server_test(self, testset):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']
        config['datasource'] = Config().data.datasource

        self.model.to(self.device)
        self.model.eval()

        custom_test = getattr(self, "testestt_model", None)

        if callable(custom_test):
            return self.test_model(config, testset)

        if config['datasource'] == 'IMDB':
            test_loader = torch.utils.data.DataLoader(
                dataset=testset,
                shuffle=False,
                batch_size=config['batch_size'],
                collate_fn=self.collate_batch,
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=config['batch_size'], shuffle=False)
        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for item in test_loader:
                examples, labels = item
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                outputs = self.model(examples)

                total_loss += loss_criterion(outputs, labels).detach().cpu().numpy()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Yield to other tasks in the server
                await asyncio.sleep(0)

            if self.current_round >= Config().trainer.rounds - 1:
                self.tsne_vis(test_loader=test_loader)

        return correct / total


    def tsne_vis(self, test_loader):
        self.model.to(self.device)
        self.model.eval()

        outputs = []
        labels = []
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(test_loader):
                x = x.to(self.device)
                output = self.model(x)
                outputs.extend(output.detach().cpu().numpy())
                labels.extend(label.numpy())

        outputs = np.reshape(np.array(outputs), (len(outputs), -1))

        tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=False,
        )

        embedding = tsne.fit(outputs)
        _, ax = plt.subplots()
        ax = plot(embedding, labels, ax=ax)
        plt.savefig('TSNE_{}.pdf'.format(self.client_id))