import logging
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

from src.utils.submission import generate_submission


class PassageMatcherPL(pl.LightningModule):
    def __init__(self,
                 cfg: DictConfig,
                 test_loader: DataLoader,
                 global_logger: logging.Logger
                 ):
        super().__init__()
        self.cfg = cfg
        self.test_loader = test_loader
        self.global_logger = global_logger
        self.encoder_config: PretrainedConfig = AutoConfig.from_pretrained(self.cfg.biencoder.model_name)
        self.passage_encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.biencoder.model_name,
                                                                          config=self.encoder_config)
        self.classifier: nn.Module = nn.Linear(self.encoder_config.hidden_size, 2)
        self.automatic_optimization = False
        self.epoch_idx = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.learning_rate,
                                      weight_decay=self.cfg.train.weight_decay)
        return optimizer

    def forward(
            self,
            query_ids: T,
            query_segments: T,
            query_attention_mask: T,
            context_ids: T,
            context_segments: T,
            context_attention_mask: T
    ):

        encoded_query = self.passage_encoder(
            input_ids=query_ids,
            token_type_ids=query_segments,
            attention_mask=query_attention_mask
        ).last_hidden_state[:, 0, :]

        encoded_context = self.passage_encoder(
            input_ids=context_ids,
            token_type_ids=context_segments,
            attention_mask=context_attention_mask
        ).last_hidden_state[:, 0, :]

        matching_logits = encoded_query * encoded_context
        matching_logits = self.classifier(matching_logits)

        return matching_logits

    def training_step(self, batch: Dict, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()

        query_passage: Dict = batch["query"]
        context_passage: Dict = batch["context"]
        label: T = batch["label"].to(self.device)
        query_ids: T = query_passage["input_ids"].squeeze(1)
        query_segments: T = query_passage["token_type_ids"].squeeze(1)
        query_attention_mask: T = query_passage["attention_mask"].squeeze(1)
        context_ids: T = context_passage["input_ids"].squeeze(1)
        context_segments: T = context_passage["token_type_ids"].squeeze(1)
        context_attention_mask: T = context_passage["attention_mask"].squeeze(1)

        matching_logits = self.forward(
            query_ids=query_ids,
            query_segments=query_segments,
            query_attention_mask=query_attention_mask,
            context_ids=context_ids,
            context_segments=context_segments,
            context_attention_mask=context_attention_mask
        )

        matching_scores = F.softmax(matching_logits, dim=1)

        loss = F.cross_entropy(matching_scores, label, reduction="mean")
        self.manual_backward(loss)
        optimizer.step()

        self.log("train_step_loss", loss)
        return {"train_step_loss": loss.detach()}

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        loss.backward()

    def on_train_epoch(self) -> None:
        self.epoch_idx += 1

    def validation_step(self, batch: Dict, batch_idx):
        query_passage: Dict = batch["query"]
        context_passage: Dict = batch["context"]
        label: T = batch["label"]
        query_ids: T = query_passage["input_ids"].squeeze(1)
        query_segments: T = query_passage["token_type_ids"].squeeze(1)
        query_attention_mask: T = query_passage["attention_mask"].squeeze(1)
        context_ids: T = context_passage["input_ids"].squeeze(1)
        context_segments: T = context_passage["token_type_ids"].squeeze(1)
        context_attention_mask: T = context_passage["attention_mask"].squeeze(1)

        matching_logits = self.forward(
            query_ids=query_ids,
            query_segments=query_segments,
            query_attention_mask=query_attention_mask,
            context_ids=context_ids,
            context_segments=context_segments,
            context_attention_mask=context_attention_mask
        )

        matching_scores = F.softmax(matching_logits, dim=1)

        loss = F.cross_entropy(matching_scores, label, reduction="mean")
        self.log("val_step_loss", loss)
        return {"val_step_loss": loss.detach()}

    def validation_epoch_end(self, outputs) -> Dict:
        avg_loss = torch.tensor([x['val_step_loss'].mean() for x in outputs]).mean()
        self.log("val_epoch_loss", avg_loss)
        return {"val_avg_loss": avg_loss}

    def on_validation_end(self) -> None:
        self.eval()
        test_results: List[np.ndarray] = []
        with torch.no_grad():
            with tqdm(range(len(self.test_loader))) as pbar:
                for idx, batch in enumerate(self.test_loader):
                    query_passage: Dict = batch["query"]
                    context_passage: Dict = batch["context"]
                    query_ids: T = query_passage["input_ids"].squeeze(1).to(self.device)
                    query_segments: T = query_passage["token_type_ids"].squeeze(1).to(self.device)
                    query_attention_mask: T = query_passage["attention_mask"].squeeze(1).to(self.device)
                    context_ids: T = context_passage["input_ids"].squeeze(1).to(self.device)
                    context_segments: T = context_passage["token_type_ids"].squeeze(1).to(self.device)
                    context_attention_mask: T = context_passage["attention_mask"].squeeze(1).to(self.device)

                    pred = self.forward(
                        query_ids=query_ids,
                        query_segments=query_segments,
                        query_attention_mask=query_attention_mask,
                        context_ids=context_ids,
                        context_segments=context_segments,
                        context_attention_mask=context_attention_mask
                    )
                    pred = F.softmax(pred, dim=1)[:, 1]
                    test_results.append(pred.detach().cpu().numpy())
                    if idx % 10 == 0:
                        pbar.update(10)
        generate_submission(f'./submissions/epoch_{self.epoch_idx}', np.concatenate(test_results))
