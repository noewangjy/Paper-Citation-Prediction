import logging
from typing import List

import dgl
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as torch_f
import torchmetrics
import tqdm
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_logger
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.models import MLPBert
from src.utils.driver import NetworkDatasetMLPBert
from src.utils.submission import generate_submission


def compute_train_loss(pred: torch.Tensor, label: torch.Tensor):
    return torch_f.binary_cross_entropy_with_logits(pred.to(torch.float32), label.to(torch.float32))


def compute_acc(pred: torch.Tensor, label: torch.Tensor):
    ans = torch.argmax(pred, dim=-1)
    acc = torchmetrics.functional.accuracy(ans, label)
    return acc


def prepare_graph(cfg,
                  tokenizer: AutoTokenizer,
                  device: torch.device = torch.device('cpu')) -> {}:
    print("Preparing graphs - Unpickling")
    graphs = {'train_set': NetworkDatasetMLPBert(to_absolute_path(cfg.train_dataset_path),
                                                 tokenizer,
                                                 cfg.author_token_length,
                                                 cfg.abstract_token_length),
              'dev_set': NetworkDatasetMLPBert(to_absolute_path(cfg.dev_dataset_path),
                                               tokenizer,
                                               cfg.author_token_length,
                                               cfg.abstract_token_length),
              'test_set': NetworkDatasetMLPBert(to_absolute_path(cfg.test_dataset_path),
                                                tokenizer,
                                                cfg.author_token_length,
                                                cfg.abstract_token_length)
              }
    print("Preparing graphs - Creating sub graphs")
    graphs['num_nodes'] = graphs['train_set'].graph.number_of_nodes()
    graphs['train_graph'] = dgl.graph((graphs['train_set'].u, graphs['train_set'].v), num_nodes=graphs['num_nodes']).to(device)
    graphs['train_pos_graph'] = dgl.graph((graphs['train_set'].pos_u, graphs['train_set'].pos_v), num_nodes=graphs['num_nodes']).to(device)
    graphs['train_neg_graph'] = dgl.graph((graphs['train_set'].neg_u, graphs['train_set'].neg_v), num_nodes=graphs['num_nodes']).to(device)
    graphs['dev_graph'] = dgl.graph((graphs['dev_set'].u, graphs['dev_set'].v), num_nodes=graphs['num_nodes']).to(device)
    graphs['dev_pos_graph'] = dgl.graph((graphs['dev_set'].pos_u, graphs['dev_set'].pos_v), num_nodes=graphs['num_nodes']).to(device)
    graphs['dev_neg_graph'] = dgl.graph((graphs['dev_set'].neg_u, graphs['dev_set'].neg_v), num_nodes=graphs['num_nodes']).to(device)

    return graphs


class MLPSolution(pl.LightningModule):
    def __init__(self,
                 hydra_cfg: DictConfig,
                 test_loader: DataLoader,
                 global_logger: logging.Logger):
        super().__init__()
        self.hydra_config = hydra_cfg
        self.test_loader = test_loader
        self.global_logger = global_logger
        self.model = MLPBert(bert_model_name=self.hydra_config.model.bert_model_name,
                             bert_num_feature=self.hydra_config.model.bert_num_feature)
        self.automatic_optimization = False
        self.save_hyperparameters(self.hydra_config)
        self.epoch_idx: int = 0
        self.accuracy_matrix = torchmetrics.Accuracy().to(self.device)
        self.skip_submission_generation: bool = True

    def forward(self, u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts):
        """
        :param
        :return:
        """
        return self.model(u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts)

    def training_step(self, batch, batch_idx):
        sample = batch
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        u, v, u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts, y = sample
        label = torch_f.one_hot(y, num_classes=2).squeeze(1)
        pred = self(u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts)

        loss = compute_train_loss(pred, label)

        self.manual_backward(loss)
        optimizer.step()
        self.log('loss', loss)
        return {'loss': loss}

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        loss.backward()

    def on_train_epoch_end(self) -> None:
        self.epoch_idx += 1
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                self.logger.experiment.add_histogram('model_params' + name, param, self.epoch_idx)
                self.log('model_params' + name, param)

    def on_train_start(self) -> None:
        pass
        # self.epoch_idx = 0

    def on_validation_epoch_start(self) -> None:
        self.accuracy_matrix.reset()

    def validation_step(self, batch, batch_idx):
        sample = batch
        u, v, u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts, y = sample
        label = torch_f.one_hot(y, num_classes=2).squeeze(1)
        pred = self(u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts)
        loss = compute_train_loss(pred, label)

        answer: torch.Tensor = torch.softmax(pred, dim=-1)
        self.accuracy_matrix(answer, label)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['val_loss'].mean() for x in outputs]).mean()
        self.log('val_avg_loss', avg_loss)

        return {'val_avg_loss': avg_loss}

    def on_validation_end(self) -> None:
        if self.skip_submission_generation:
            self.skip_submission_generation = False
            with torch.no_grad():
                try:
                    test_result: List[np.ndarray] = []
                    test_u: List[np.ndarray] = []
                    with torch.no_grad():
                        with tqdm.tqdm(range(len(self.test_loader))) as pbar:
                            for idx, sample in enumerate(self.test_loader):
                                u, v, u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts, _ = map(lambda x: (torch.unsqueeze(x, 0) if len(x.shape) == 1 else x).to(self.device), sample)
                                pred = self(u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts)
                                pred_softmax = torch.softmax(pred, dim=-1)
                                test_result.append(pred_softmax[:, 1].detach().cpu().numpy())
                                test_u.append(u.detach().cpu().numpy())
                                if idx % 10 == 0: pbar.update(10)

                    generate_submission(f'./submissions/epoch_{self.epoch_idx}', np.concatenate(test_result))
                    # generate_submission(f'./epoch{self.epoch_idx}_index', np.concatenate(test_u))
                    np.savetxt(f'./submissions/epoch_{self.epoch_idx}/index.txt', np.concatenate(test_u), delimiter=', ', fmt="%d")
                except Exception as e:
                    self.global_logger.error(e)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hydra_config.train.lr, weight_decay=0.)


@hydra.main(config_path="conf", config_name="config")
def run(cfg):
    # >>>>>> Beg Configuration <<<<<<
    SEED = 2022
    DEBUG: bool = cfg.debug
    DEVICE = torch.device('cpu')
    print("Preparing tokenizer - Loading vocabs")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model_name)
    # <<<<<< End Configuration <<<<<<

    # >>>>>> Beg Init >>>>>>
    torch.manual_seed(SEED)
    graphs = prepare_graph(cfg.dataset,
                           tokenizer,
                           DEVICE)
    train_set = Subset(graphs['train_set'], range(1000)) if DEBUG else graphs['train_set']
    dev_set = Subset(graphs['dev_set'], range(100)) if DEBUG else graphs['dev_set']
    test_set = Subset(graphs['test_set'], range(1000)) if DEBUG else graphs['test_set']

    train_loader = DataLoader(train_set,
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=cfg.machine.num_workers,
                              persistent_workers=True,
                              drop_last=True)

    dev_loader = DataLoader(dev_set,
                            batch_size=cfg.train.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=cfg.machine.num_workers,
                            persistent_workers=True,
                            drop_last=True)

    test_loader = DataLoader(test_set,
                             batch_size=cfg.test.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=cfg.machine.num_workers,
                             persistent_workers=True,
                             drop_last=False)

    logger = pl_logger.TensorBoardLogger(save_dir=cfg.io.log_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints",
                                                       filename="{epoch}-{step}-{val_avg_loss:.4f}",
                                                       monitor='val_avg_loss',
                                                       save_last=True,
                                                       save_top_k=cfg.io.num_checkpoints,
                                                       mode='min',
                                                       save_weights_only=True,
                                                       every_n_train_steps=cfg.io.every_n_train_steps,
                                                       save_on_train_epoch_end=True)
    trainer = pl.Trainer(gpus=cfg.machine.gpus if not DEBUG else [cfg.machine.gpus[0]],
                         max_epochs=cfg.train.max_epochs,
                         callbacks=[checkpoint_callback,
                                    TQDMProgressBar(refresh_rate=10)],
                         enable_checkpointing=True,
                         val_check_interval=cfg.io.val_check_interval,
                         # progress_bar_refresh_rate=10,
                         default_root_dir=cfg.io.checkpoint_dir,
                         logger=logger)
    global_logger = logging.getLogger(__name__)
    global_logger.info("Start training")
    solution = MLPSolution(cfg, test_loader, global_logger)
    # solution.on_validation_end()
    trainer.fit(solution, train_loader, dev_loader)
    # <<<<<< Beg Init <<<<<<


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    run()
