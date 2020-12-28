import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset


def binary_acc(pred, y):
    rounded_pred = torch.round(pred)
    correct = (rounded_pred == y).float()
    return correct.sum() / len(correct)


class Config:
    num_embeddings = 10000
    embedding_dim = 16
    hidden_size = 64
    sequence_length = 512
    num_layers = 1
    bidirectional = True
    output_size = 1
    batch_size = 64
    num_dataloader_workers = 2
    learning_rate = 1e-3


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_embeddings = config.num_embeddings
        self.sequence_length = config.sequence_length

    def prepare_data(self):
        # Download imdb dataset
        train_full = load_dataset("imdb", split="train").train_test_split(test_size=0.1)
        train = train_full["train"]
        val = train_full["test"]
        test = load_dataset("imdb", split="test").shuffle(seed=42)

        # Convert dataset into a list of (text, label) pairs
        train_text = [i["text"] for i in train]
        train_label = np.array([i["label"] for i in train])
        val_text = [i["text"] for i in val]
        val_label = np.array([i["label"] for i in val])
        test_text = [i["text"] for i in test]
        test_label = np.array([i["label"] for i in test])

        # Make tokenizer
        tokenizer = Tokenizer(num_words=self.num_embeddings, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_full["train"]["text"] + train_full["test"]["text"])

        # Convert text to integer sequences and pad sequences to the same length
        train_text = pad_sequences(
            tokenizer.texts_to_sequences(train_text),
            maxlen=self.sequence_length,
            truncating="post",
            padding="post",
        )
        val_text = pad_sequences(
            tokenizer.texts_to_sequences(val_text),
            maxlen=self.sequence_length,
            truncating="post",
            padding="post",
        )
        test_text = pad_sequences(
            tokenizer.texts_to_sequences(test_text),
            maxlen=self.sequence_length,
            truncating="post",
            padding="post",
        )

        # Convert numpy arrays to tensors
        train_text, train_label = map(torch.from_numpy, [train_text, train_label])
        val_text, val_label = map(torch.from_numpy, [val_text, val_label])
        test_text, test_label = map(torch.from_numpy, [test_text, test_label])

        # Make tensor datasets
        self.train = TensorDataset(train_text, train_label)
        self.val = TensorDataset(val_text, val_label)
        self.test = TensorDataset(test_text, test_label)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
        )


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.hidden_size = config.hidden_size
        self.sequence_length = config.sequence_length
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.output_size = config.output_size
        self.learning_rate = config.learning_rate

        self.embedding = nn.Embedding(
            num_embeddings=config.num_embeddings + 1, embedding_dim=config.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        self.linear = nn.Linear(config.hidden_size * 2, config.output_size)

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
            weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_(),
            weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_(),
        )
        return hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y.float())
        acc = binary_acc(output, y.float())
        self.log("train_loss", loss, logger=True)
        self.log("train_acc", acc, logger=True)
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([i["loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        print(f"Epoch {self.current_epoch} average training loss: {avg_loss}")
        print(f"Epoch {self.current_epoch} average training accuracy: {avg_acc}\n")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y.float())
        acc = binary_acc(output, y.float())
        self.log("val_loss", loss, logger=True)
        self.log("val_acc", acc, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        print(f"Epoch {self.current_epoch} average validation loss: {avg_loss}")
        print(f"Epoch {self.current_epoch} average validation accuracy: {avg_acc}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y.float())
        acc = binary_acc(output, y.float())
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        print(f"Average test loss: {avg_loss}")
        print(f"Average test accuracy: {avg_acc}")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


config = Config()
datamodule = DataModule(config)
datamodule.prepare_data()
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()

early_stop_callback = EarlyStopping(
    monitor="val_acc", patience=10, verbose=True, mode="max"
)
logger = TensorBoardLogger("tb_logs", name="model")

model = Model(config)
trainer = Trainer(
    gpus=1, progress_bar_refresh_rate=50, callbacks=[early_stop_callback], logger=logger
)
trainer.fit(
    model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
)

test_result = trainer.test(model=model, test_dataloaders=test_dataloader)

# %load_ext tensorboard
# %tensorboard --logdir tb_logs/
