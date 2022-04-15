import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.submission import generate_submission


class EmbeddingClassifier(pl.LightningModule):
    def __init__(self,
                 config: DictConfig,
                 test_loader: DataLoader
                 ):
        super().__init__()
        self.epoch_idx = 0
        self.config = config
        self.test_loader = test_loader
        self.embedding = nn.Embedding(config.model.vocab_size, config.model.embedding_dim)
        self.classifier = nn.Linear(config.model.embedding_dim, 2)
        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(config.model.embedding_dim),
        #     nn.Linear(config.model.embedding_dim, config.model.hidden_size1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(config.model.hidden_size1),
        #     nn.Linear(config.model.hidden_size1, config.model.hidden_size2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(config.model.hidden_size2),
        #     nn.Linear(config.model.hidden_size2, 2)
        # )

        self.automatic_optimization = False

    def forward(self, u: T, v: T):
        # u_embedding.size() = [batch_size, embedding_dim]
        u_embedding = self.embedding(u)
        v_embedding = self.embedding(v)
        # TODO: Modify this part
        embedding = u_embedding + v_embedding
        out = self.classifier(embedding)
        # End of modification
        scores = F.softmax(out, dim=1)
        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.train.learning_rate,
                                     weight_decay=self.config.train.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        u = batch["u"]
        v = batch["v"]
        y = batch["y"]
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        pred = self(u, v)
        loss = F.cross_entropy(pred, y, reduction="mean")
        self.manual_backward(loss)
        optimizer.step()
        self.log("train_step_loss", loss)
        return {'train_step_loss': loss.detach()}

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        loss.backward()

    def on_train_start(self) -> None:
        pass
        # self.epoch_idx = 0

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        for name, param in self.named_parameters():
            if 'bn' not in name:
                self.log('model_params_' + name, param)
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        u = batch["u"]
        v = batch["v"]
        y = batch["y"]
        pred = self(u, v)
        loss = F.cross_entropy(pred, y, reduction="mean")
        self.log('val_step_loss', loss)
        return {'val_step_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x["val_step_loss"].mean() for x in outputs]).mean()
        self.log("val_epoch_loss", avg_loss)
        return {"val_epoch_loss": avg_loss}

    def on_validation_end(self) -> None:
        self.eval()
        test_results = []
        with torch.no_grad():
            with tqdm(range(len(self.test_loader))) as pbar:
                for idx, sample in enumerate(self.test_loader):
                    u = sample["u"].to(self.device)
                    v = sample["v"].to(self.device)
                    y = sample["y"].to(self.device)
                    pred = self(u, v)
                    test_results.extend(pred[:, 1].detach().cpu().numpy())
                    if idx % 10 == 0:
                        pbar.update(10)
        generate_submission(f"./submissions/epoch_{self.epoch_idx}", np.array(test_results))
