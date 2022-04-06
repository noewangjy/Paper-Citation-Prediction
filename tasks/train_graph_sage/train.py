import dgl
import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as torch_f
import torchmetrics
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_logger
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import tqdm

from src.models import GraphSAGEBundled
from src.utils.driver import NetworkDatasetGraphSAGEBert


def compute_train_loss(pos_score: torch.Tensor, neg_score: torch.Tensor):
    device = pos_score.device
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).to(device)
    loss = torch_f.binary_cross_entropy(scores, labels)
    return loss

@torch.no_grad()
def compute_acc(pos_score: torch.Tensor, neg_score: torch.Tensor):
    device = pos_score.device
    scores = torch.cat([pos_score, neg_score])
    scores = torch.greater(scores, 0.5)
    labels = torch.cat([torch.ones_like(pos_score, dtype=torch.int32), torch.zeros_like(neg_score, dtype=torch.int32)]).to(device)
    acc = torchmetrics.functional.accuracy(scores, labels)
    return acc


# def compute_auc(pos_score: torch.Tensor, neg_score: torch.Tensor):
#     scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
#     labels = torch.cat(
#         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().cpu().numpy()
#     return roc_auc_score(labels, scores)


def prepare_graph(cfg,
                  device: torch.device = torch.device('cpu')) -> {}:
    print("Preparing graphs - Unpickling")
    datasets = {'train_set': NetworkDatasetGraphSAGEBert(to_absolute_path(cfg.train_dataset_path),
                                                         to_absolute_path(cfg.features_path)),
                'dev_set': NetworkDatasetGraphSAGEBert(to_absolute_path(cfg.dev_dataset_path),
                                                       to_absolute_path(cfg.features_path)),
                'test_set': NetworkDatasetGraphSAGEBert(to_absolute_path(cfg.test_dataset_path),
                                                        to_absolute_path(cfg.features_path))
                }
    print("Preparing graphs - Creating sub graphs")
    num_nodes = datasets['train_set'].graph.number_of_nodes()
    num_features = datasets['train_set'].data['node_features'].shape[1]
    graphs = {'features': datasets['train_set'].data['node_features'].to(device),
              'num_nodes': num_nodes,
              'num_features': num_features,
              'train_graph': dgl.graph((datasets['train_set'].u, datasets['train_set'].v), num_nodes=num_nodes).to(device),
              'train_pos_graph': dgl.graph((datasets['train_set'].pos_u, datasets['train_set'].pos_v), num_nodes=num_nodes).to(device),
              'train_neg_graph': dgl.graph((datasets['train_set'].neg_u, datasets['train_set'].neg_v), num_nodes=num_nodes).to(device), 'dev_graph': dgl.graph((datasets['dev_set'].u, datasets['dev_set'].v), num_nodes=num_nodes).to(device),
              'dev_pos_graph': dgl.graph((datasets['dev_set'].pos_u, datasets['dev_set'].pos_v), num_nodes=num_nodes).to(device), 'dev_neg_graph': dgl.graph((datasets['dev_set'].neg_u, datasets['dev_set'].neg_v), num_nodes=num_nodes).to(device),
              'test_graph': dgl.graph((datasets['test_set'].u, datasets['test_set'].v), num_nodes=num_nodes).to(device),
              'test_set': datasets['test_set']}
    return graphs


class GraphSolution(pl.LightningModule):
    def __init__(self, hydra_cfg: DictConfig, graphs):
        super().__init__()
        self.hydra_config = hydra_cfg
        self.graphs = graphs
        self.model = GraphSAGEBundled(self.graphs['num_features'],
                                      hydra_cfg.model.hidden_dims,
                                      hydra_cfg.model.predictor,
                                      hydra_cfg.model.aggregator)
        self.automatic_optimization = False
        self.save_hyperparameters(self.hydra_config)
        self.epoch_idx = 0
        self.accuracy_matrix = torchmetrics.Accuracy().to(self.device)
        self.pbar = tqdm.tqdm(range(hydra_cfg.train.max_epochs))

    def forward(self, g, g_features):
        """
        :param
        :return:
        """
        return self.model(g, g_features)

    def training_step(self, *args):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()

        h = self(self.graphs['train_graph'], self.graphs['features'])
        pos_score = self.model.predictor(self.graphs['train_pos_graph'], h)
        neg_score = self.model.predictor(self.graphs['train_neg_graph'], h)

        loss = compute_train_loss(pos_score, neg_score)

        self.manual_backward(loss)
        optimizer.step()
        return {'loss': loss}

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        loss.backward()

    def on_train_epoch_end(self) -> None:
        self.epoch_idx += 1
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                self.logger.experiment.add_histogram('model_params' + name, param, self.epoch_idx)

    def on_train_start(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        self.accuracy_matrix.reset()

    def validation_step(self, *args):
        if self.model.hidden_state is not None:
            pos_score = self.model.predictor(self.graphs['dev_pos_graph'], self.model.hidden_state)
            neg_score = self.model.predictor(self.graphs['dev_neg_graph'], self.model.hidden_state)
            acc = compute_acc(pos_score, neg_score)
            loss = compute_train_loss(pos_score, neg_score)
            self.pbar.set_description(f"epoch_idx: {self.epoch_idx}, acc: {acc.detach().cpu().numpy()}, loss: {loss.detach().cpu().numpy()}")
            self.pbar.update(1)
            return {'val_acc': acc, 'val_loss': loss}
        else:
            return {'val_acc': torch.tensor([1.]), 'val_loss': torch.tensor([1.])}

    def validation_epoch_end(self, outputs):
        avg_acc = torch.tensor([x['val_acc'].mean() for x in outputs]).mean()
        avg_loss = torch.tensor([x['val_loss'].mean() for x in outputs]).mean()

        self.log('val_avg_acc', avg_acc)
        self.log('val_avg_loss', avg_loss)

        return {'val_avg_acc': avg_acc, 'val_avg_loss': avg_loss}

    def train_dataloader(self):
        return torch.tensor([1])

    def val_dataloader(self):
        return torch.tensor([1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hydra_config.train.lr, weight_decay=0.)


@hydra.main(config_path="conf", config_name="config")
def run(cfg):
    # >>>>>> Beg Configuration <<<<<<
    DEVICE = torch.device('cpu') if len(cfg.machine.gpus) <= 0 else torch.device(f'cuda:{cfg.machine.gpus[0]}')
    # print("Preparing tokenizer - Loading vocabs")
    # tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model_name)
    # <<<<<< End Configuration <<<<<<

    # >>>>>> Beg Init >>>>>>
    graphs = prepare_graph(cfg.dataset,
                           DEVICE)
    # train_loader = DataLoader(graphs['train_set'],
    #                           batch_size=cfg.train.batch_size,
    #                           shuffle=True,
    #                           pin_memory=True,
    #                           num_workers=cfg.machine.num_workers,
    #                           persistent_workers=True)
    #
    # dev_loader = DataLoader(graphs['dev_set'],
    #                         batch_size=cfg.train.batch_size,
    #                         shuffle=False,
    #                         pin_memory=True,
    #                         num_workers=cfg.machine.num_workers,
    #                         persistent_workers=True)

    logger = pl_logger.TensorBoardLogger(save_dir=cfg.io.log_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints",
                                                       filename="{epoch}-{val_avg_loss:.4f}",
                                                       monitor='val_avg_loss',
                                                       save_last=True,
                                                       save_top_k=cfg.io.num_checkpoints,
                                                       mode='min',
                                                       save_weights_only=False,
                                                       every_n_epochs=cfg.io.save_interval,
                                                       save_on_train_epoch_end=True)
    trainer = pl.Trainer(gpus=cfg.machine.gpus,
                         max_epochs=cfg.train.max_epochs,
                         callbacks=[checkpoint_callback],
                         enable_checkpointing=True,
                         progress_bar_refresh_rate=0,
                         default_root_dir=cfg.io.checkpoint_dir,
                         logger=logger,
                         log_every_n_steps=1)
    solution = GraphSolution(cfg, graphs)
    trainer.fit(solution)
    # <<<<<< Beg Init <<<<<<


if __name__ == '__main__':
    run()
