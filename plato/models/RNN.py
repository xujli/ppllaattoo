import torch
import torch.nn as nn

from plato.config import Config

class Model(nn.Module):
    def __init__(self, input_dim=100683, embedding_dim=4, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 8)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        lstm, _ = self.lstm(embedded)

        # embedded = [sent len, batch size, emb dim]
        fc = self.fc(lstm[:, :-1, :])
        return fc


    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        if hasattr(Config().trainer, 'num_classes'):
            return Model(num_classes=Config().trainer.num_classes)
        return Model()