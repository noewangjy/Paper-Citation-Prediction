import logging
import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig


class EmbeddingClassifier(pl.LightningModule):
    def __init__(self,
                 config: DictConfig,
                 global_logger: logging.Logger
                 ):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.model.vocab_size, config.model.embedding_dim)
        self.linear1 = nn.Linear(config.model.embedding_dim, config.model.hidden_size)
        self.linear2 = nn.Linear(config.model.hidden_size, 2)
        self.global_logger = global_logger

    def forward(self, input_ids: T):
        embedding = self.embedding(input_ids)
        out = self.linear2(F.relu(self.linear1(embedding)))
        scores = F.softmax(out, dim=1)
        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.train.learning_rate,
                                     weight_decay=self.config.train.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        nodes = batch["nodes"]
        label = batch["label"]
        # label = F.one_hot(label, num_classes=2).squeeze(1)
        pred = self(nodes)
        loss = F.cross_entropy(pred, label, reduction="mean")
        self.manual_backward(loss)
        self.log("train_loss", loss)
        return {'loss': loss}

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        loss.backward()

    def on_train_start(self) -> None:
        pass
        # self.epoch_idx = 0

    def on_train_epoch_end(self) -> None:
        self.epoch_idx += 1
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                self.log('model_params' + name, param)

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        nodes = batch["nodes"]
        label = batch["label"]
        pred = self(nodes)
        loss = F.cross_entropy(pred, label, reduction="mean")
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x["val_loss"].mean() for x in outputs]).mean()
        self.log("val_avg_loss", avg_loss)
        return {"val_avg_loss": avg_loss}

    def on_validation_end(self) -> None:
        pass











