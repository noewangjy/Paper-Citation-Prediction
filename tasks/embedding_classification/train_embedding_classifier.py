import logging

import pytorch_lightning as pl
import torch.backends.cudnn

from src.utils import NetworkDatasetEmbeddingClassification
import hydra
from torch.utils.data import Subset, DataLoader
import numpy as np
from src.models.embedding_classification.modeling import EmbeddingClassifier
from hydra.utils import to_absolute_path


@hydra.main(config_path="conf", config_name="config")
def main(args):
    train_dataset = NetworkDatasetEmbeddingClassification(to_absolute_path(args.data.train_file))
    if args.data.train_size:
        train_dataset = Subset(train_dataset, np.arange(args.data.train_size)-int(args.data.train_size))
    dev_dataset = NetworkDatasetEmbeddingClassification(to_absolute_path(args.data.dev_file))
    if args.data.dev_size:
        dev_dataset = Subset(dev_dataset, np.arange(args.data.dev_size) - int(args.data.dev_size))

    test_dataset = NetworkDatasetEmbeddingClassification(to_absolute_path(args.data.test_file))

    args.model.vocab_size = len(train_dataset.dataset.abstracts) if hasattr(train_dataset, "dataset") else len(train_dataset.abstracts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train.batch_size,
        shuffle=True,
        # pin_memory=True,
        num_workers=args.train.num_workers,
        persistent_workers=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.train.batch_size,
        shuffle=False,
        # pin_memory=True,
        num_workers=args.train.num_workers,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.train.batch_size,
        shuffle=False,
        num_workers=args.train.num_workers,
        persistent_workers=True
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=args.train.log_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}--{step}--{val_epoch_loss:.4f}",
        monitor="val_epoch_loss",
        save_last=True,
        save_top_k=args.train.num_checkpoints,
        mode="min",
        save_weights_only=True,
        every_n_train_steps=args.train.every_n_train_steps,
        save_on_train_epoch_end=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.machine.gpus,
        max_epochs=args.train.num_epochs,
        callbacks=[checkpoint_callback,
                   pl.callbacks.TQDMProgressBar(refresh_rate=10)],
        enable_checkpointing=True,
        # val_check_interval=args.train.val_check_interval,
        default_root_dir=args.model.checkpoint_path,
        logger=logger,
        num_sanity_val_steps=0,
    )

    global_logger = logging.getLogger(__name__)
    global_logger.info("Start training")
    model = EmbeddingClassifier(args, test_loader=test_loader)
    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()



