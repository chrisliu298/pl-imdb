import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class Class:
        pass

    obj = Class()
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
    return obj


class DataModule:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_embeddings = config.num_embeddings
        self.seq_len = config.seq_len

    def prepare_data(self):
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
        tokenizer = Tokenizer(num_words=self.num_embeddings, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_full["train"]["text"] + train_full["test"]["text"])

        # Convert text to integer sequences and pad sequences to the same length
        train_text = pad_sequences(
            tokenizer.texts_to_sequences(train_text),
            maxlen=self.seq_len,
            truncating="post",
            padding="post",
        )
        val_text = pad_sequences(
            tokenizer.texts_to_sequences(val_text),
            maxlen=self.seq_len,
            truncating="post",
            padding="post",
        )
        test_text = pad_sequences(
            tokenizer.texts_to_sequences(test_text),
            maxlen=self.seq_len,
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


class Model(nn.Module):
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
        out = torch.sigmoid(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_(),
            weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_(),
        )
        return hidden


config = {
    "num_embeddings:": 10000,
    "embedding_dim": 10,
    "hidden_size": 64,
    "seq_len": 512,
    "num_layers": 1,
    "bidirectional": True,
    "output_size": 1,
    "batch_size": 64,
}

data_module = DataModule(config)
