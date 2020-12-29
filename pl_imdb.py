import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data import DataModule
from model import BiLSTM

datamodule = DataModule("imdb_train.tsv", "imdb_test.tsv")
datamodule.prepare_data()
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()

early_stop_callback = EarlyStopping(
    monitor="val_acc", patience=10, verbose=True, mode="max"
)
logger = TensorBoardLogger("tb_logs", name="model")
model = BiLSTM()
trainer = Trainer(
    gpus=1,
    progress_bar_refresh_rate=50,
    callbacks=[early_stop_callback],
    logger=logger,
)
trainer.fit(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
test_result = trainer.test(model=model, test_dataloaders=test_dataloader)
