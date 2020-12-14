import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset


def dict_to_obj(d):
    if isinstance(d, list):
        d = [dict_to_obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class Class:
        pass

    obj = Class()
    for k in d:
        obj.__dict__[k] = dict_to_obj(d[k])
    return obj


def binary_acc(pred, y):
    rounded_pred = torch.round(pred)
    correct = (rounded_pred == y).float()
    return correct.sum() / len(correct)


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


# Define configuration
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
config = dict_to_obj(config)

# Create data module and prepare data
data_module = DataModule(config)
data_module.prepare_data()

# Make data loaders for each split
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Create model
model = Model(config)
model.cuda()

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create criterion
criterion = nn.BCELoss()

# Define other training parameters
epochs = 10
print_interval = 1000
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.train()
for epoch in range(epochs):
    h = model.init_hidden(config.batch_size)
    train_losses = []
    train_accs = []
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        h = tuple([i.data for i in h])

        for p in model.parameters():
            p.grad = None  # Equivalent to model.zero_grad() but better

        output, h = model(x, h)
        acc = binary_acc(output, y.float())
        loss = criterion(output, y.float())

        train_accs.append(acc.item())
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch_idx % print_interval == 0 or batch_idx == len(train_loader):
            model.eval()
            with torch.no_grad():
                val_h = model.init_hidden(config.batch_size)
                val_losses = []
                val_accs = []
                for batch in val_loader:
                    x, y = batch
                    x, y = x.cuda(), y.cuda()
                    val_h = tuple([i.data for i in val_h])

                    output, val_h = model(x, val_h)

                    val_acc = binary_acc(output, y.float())
                    val_loss = criterion(output, y.float())

                    val_accs.append(val_acc.item())
                    val_losses.append(val_loss.item())

            model.train()
            print(
                "Epoch: {}/{}".format(epoch, epochs),
                "Step: {}".format(batch_idx),
                "Train loss: {}".format(np.mean(train_losses.item())),
                "Train acc: {}".format(np.mean(train_accs.item())),
                "Val loss: {}".format(np.mean(val_losses.item())),
                "Val acc: {}".format(np.mean(val_accs.item())),
            )


test_losses = []
test_accs = []
test_h = model.init_hidden(config.batch_size)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x, y = x.cuda(), y.cuda()
        test_h = tuple([i.data for i in test_h])

        output, test_h = model(x, test_h)

        test_acc = binary_acc(output, y.float())
        test_loss = criterion(output, y.float())

        test_accs.append(test_acc.item())
        test_losses.append(test_losses.item())

print(
    "Test loss: {}".format(np.mean(test_losses)),
    "Test acc: {}".format(np.mean(test_accs)),
)
