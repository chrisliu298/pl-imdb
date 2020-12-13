import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Download imdb dataset
        train_full = load_dataset("imdb", split="train").train_test_split(test_size=0.2)
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
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_full["train"]["text"] + train_full["test"]["text"])

        # Convert text to integer sequences and pad sequences to the same length
        train_text = pad_sequences(
            tokenizer.texts_to_sequences(train_text),
            maxlen=512,
            truncating="post",
            padding="post",
        )
        val_text = pad_sequences(
            tokenizer.texts_to_sequences(val_text),
            maxlen=512,
            truncating="post",
            padding="post",
        )
        test_text = pad_sequences(
            tokenizer.texts_to_sequences(test_text),
            maxlen=512,
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
            self.train, batch_size=self.batch_size, num_workers=2, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=2, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=2, pin_memory=True
        )


class Model(pl.LightningModule):
    def __init__(self, config):
        self.config = config
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.hidden_size = config.hidden_size
        self.seq_len = config.seq_len
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.output_size = config.output_size

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.seq_len, 2, self.hidden_size)

        lstm_out_backward = lstm_out[:, 0, 1, :]
        lstm_out_forward = lstm_out[:, -1, 0, :]

        lstm_out = torch.cat((lstm_out_backward, lstm_out_forward), dim=1)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

        out = torch.relu(self.linear(lstm_out))
        return out, hidden

    def training_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
