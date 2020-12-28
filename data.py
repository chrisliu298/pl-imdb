import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_embeddings=10000,
        sequence_length=512,
        num_dataloader_workers=2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_embeddings = num_embeddings
        self.sequence_length = sequence_length
        self.num_dataloader_workers = num_dataloader_workers

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
            num_workers=self.num_dataloader_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=True,
        )
