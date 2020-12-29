import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.metrics.functional.classification import accuracy


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y.float())
        acc = accuracy(torch.round(output), y.float())
        self.log("train_loss", loss, logger=True)
        self.log("train_acc", acc, logger=True)
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([i["loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        print(
            "---\nEpoch {} average training loss: {:12.4f}\n"
            "Epoch {} average training accuracy: {:8.4f}\n---".format(
                self.current_epoch, avg_loss, self.current_epoch, avg_acc
            )
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y.float())
        acc = accuracy(torch.round(output), y.float())
        self.log("val_loss", loss, logger=True)
        self.log("val_acc", acc, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        print(
            "---\nEpoch {} average validation loss: {:10.4f}\n"
            "Epoch {} average validation accuracy: {:6.4f}\n---".format(
                self.current_epoch, avg_loss, self.current_epoch, avg_acc
            )
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y.float())
        acc = accuracy(torch.round(output), y.float())
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class BiLSTM(BaseModel):
    def __init__(
        self,
        num_embeddings=10000,
        embedding_dim=16,
        hidden_size=64,
        num_layers=1,
        bidirectional=True,
        output_size=1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # (batch_size, sequence_length)
        x = x.long()
        # (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)
        # (batch_size, sequence_length, num_directions * hidden_size)
        lstm_out, hidden = self.lstm(embedded, hidden)
        # (batch_size, sequence_length, num_directions, hidden_size)
        lstm_out = lstm_out.contiguous().view(
            -1, self.sequence_length, 2, self.hidden_size
        )
        # (batch_size, num_directions * hidden_size / 2)
        lstm_out_backward = lstm_out[:, 0, 1, :]
        # (batch_size, num_directions * hidden_size / 2)
        lstm_out_forward = lstm_out[:, -1, 0, :]
        # (batch_size, num_directions * hidden_size)
        lstm_out = torch.cat((lstm_out_backward, lstm_out_forward), dim=1)
        # (batch_size, 1)
        # pooled = self.pooling(lstm_out)
        # (batch_size, 1)
        out = torch.relu(self.linear(lstm_out))
        out = torch.squeeze(torch.sigmoid(out))
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(
                self.num_layers * 2, batch_size, self.hidden_size
            ).zero_(),
            weight.new(
                self.num_layers * 2, batch_size, self.hidden_size
            ).zero_(),
        )
        return hidden


class FCNN(BaseModel):
    def __init__(
        self,
        num_embeddings=10000,
        embedding_dim=16,
        hidden_size=64,
        sequence_length=512,
        output_size=1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.flatten = nn.Flatten()
        self.linear_mid = nn.Linear(
            sequence_length * embedding_dim, hidden_size
        )
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # (batch_size, sequence_length)
        x = x.long()
        # (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)
        # (batch_size, sequence_length * embedding_dim)
        flattened = self.flatten(embedded)
        # (batch_size, hidden_size)
        linear_mid_out = torch.relu(self.linear_mid(flattened))
        # (batch_size, output_size)
        out = torch.squeeze(torch.sigmoid(self.linear_out(linear_mid_out)))
        return out
