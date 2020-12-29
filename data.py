import pandas as pd
import pytorch_lightning as pl
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file,
        test_file,
        batch_size=64,
        num_embeddings=10000,
        sequence_length=512,
        num_dataloader_workers=2,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_embeddings = num_embeddings
        self.sequence_length = sequence_length
        self.num_dataloader_workers = num_dataloader_workers
        self.tokenizer = Tokenizer(num_words=num_embeddings, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(
            pd.read_csv(train_file, delimiter="\t")["text"]
        )

    def prepare_data(self):
        # Download imdb dataset
        train_data = pd.read_csv(self.train_file, delimiter="\t").sample(frac=1)
        test_data = pd.read_csv(self.test_file, delimiter="\t").sample(frac=1)

        train_text = train_data["text"].iloc[:20000]
        train_label = train_data["label"].iloc[:20000].to_numpy()
        val_text = train_data["text"].iloc[20000:]
        val_label = train_data["label"].iloc[20000:].to_numpy()
        test_text = test_data["text"]
        test_label = test_data["label"].to_numpy()

        train_text = self.tokenize(train_text)
        val_text = self.tokenize(val_text)
        test_text = self.tokenize(test_text)

        # Convert numpy arrays to tensors
        train_text, train_label = map(
            torch.from_numpy, [train_text, train_label]
        )
        val_text, val_label = map(torch.from_numpy, [val_text, val_label])
        test_text, test_label = map(torch.from_numpy, [test_text, test_label])

        # Make tensor datasets
        self.train = TensorDataset(train_text, train_label)
        self.val = TensorDataset(val_text, val_label)
        self.test = TensorDataset(test_text, test_label)

    def tokenize(self, sequences):
        return pad_sequences(
            self.tokenizer.texts_to_sequences(sequences),
            maxlen=self.sequence_length,
            truncating="post",
            padding="post",
        )

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
