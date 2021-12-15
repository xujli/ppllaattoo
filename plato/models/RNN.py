import torch
import torch.nn as nn

from plato.config import Config

class Model(nn.Module):
    def __init__(self, input_dim=100683, embedding_dim=100, num_classes=2):
        super().__init__()

        self.embedding = nn.EmbeddingBag(input_dim, embedding_dim, sparse=True)

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, text, offset):
        # text = [sent len, batch size]
        embedded = self.embedding(text, offset)

        # embedded = [sent len, batch size, emb dim]

        return self.fc(embedded)


    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        if hasattr(Config().trainer, 'num_classes'):
            return Model(num_classes=Config().trainer.num_classes)
        return Model()