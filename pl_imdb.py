import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, hidden):
        return

    def training_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        return

    def test_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
