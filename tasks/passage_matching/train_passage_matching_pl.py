import logging
import os

import hydra
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import Subset, DataLoader
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    PreTrainedTokenizer
)

from src.models.passage_matching.biencoder_pl import PassageMatcherPL
from src.utils import NetworkDatasetPassageMatchingPL


@hydra.main(config_path="conf_pl", config_name="config")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    tokenizer_config: PretrainedConfig = AutoConfig.from_pretrained(cfg.biencoder.model_name)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.biencoder.model_name, config=tokenizer_config)

    cfg.data.data_path = to_absolute_path(cfg.data.data_path)
    train_set = NetworkDatasetPassageMatchingPL(
        dataset_path=os.path.join(cfg.data.data_path, cfg.data.train_file),
        tokenizer=tokenizer,
        max_seq_len=cfg.biencoder.max_seq_len
    )
    if cfg.data.train_size:
        train_set = Subset(train_set, np.arange(cfg.data.train_size) - int(cfg.data.train_size / 2))
    dev_set = NetworkDatasetPassageMatchingPL(
        dataset_path=os.path.join(cfg.data.data_path, cfg.data.dev_file),
        tokenizer=tokenizer,
        max_seq_len=cfg.biencoder.max_seq_len
    )
    if cfg.data.dev_size:
        dev_set = Subset(dev_set, np.arange(cfg.data.dev_size) - int(cfg.data.dev_size / 2))
    test_set = NetworkDatasetPassageMatchingPL(
        dataset_path=os.path.join(cfg.data.data_path, cfg.data.test_file),
        tokenizer=tokenizer,
        max_seq_len=cfg.biencoder.max_seq_len
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
        drop_last=True
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
        drop_last=False
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=cfg.log_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}--{step}--{val_epoch_loss:.4f}",
        monitor="val_epoch_loss",
        save_last=True,
        save_top_k=3,
        mode="min",
        save_weights_only=True,
        save_on_train_epoch_end=True
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.epochs,
        progress_bar_refresh_rate=1,
        callbacks=[checkpoint_callback,
                   pl.callbacks.TQDMProgressBar(refresh_rate=1)],
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        logger=logger
    )
    global_logger = logging.getLogger(__name__)
    global_logger.info("Start training")
    model = PassageMatcherPL(
        cfg=cfg,
        test_loader=test_loader,
        global_logger=global_logger
    )
    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run()
