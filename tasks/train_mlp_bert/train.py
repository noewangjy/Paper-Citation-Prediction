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

from src.models import MLPBert
from src.utils.driver import NetworkDatasetSAGEBert


# def compute_loss(pos_score: torch.Tensor, neg_score: torch.Tensor):
#     device = pos_score.device
#     scores = torch.cat([pos_score, neg_score])
#     labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
#     return F.binary_cross_entropy_with_logits(scores, labels)

def compute_train_loss(pred: torch.Tensor, label: torch.Tensor):
    return torch_f.binary_cross_entropy_with_logits(pred.to(torch.float32), label.to(torch.float32))


def compute_acc(pred: torch.Tensor, label: torch.Tensor):
    ans = torch.argmax(pred, dim=-1)
    acc = torchmetrics.functional.accuracy(ans, label)
    return acc


# def compute_auc(pos_score: torch.Tensor, neg_score: torch.Tensor):
#     scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
#     labels = torch.cat(
#         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().cpu().numpy()
#     return roc_auc_score(labels, scores)


def prepare_graph(cfg,
                  tokenizer: AutoTokenizer,
                  device: torch.device = torch.device('cpu')) -> {}:
    print("Preparing graphs - Unpickling")
    graphs = {'train_set': NetworkDatasetSAGEBert(to_absolute_path(cfg.train_dataset_path),
                                                  tokenizer,
                                                  cfg.author_token_length,
                                                  cfg.abstract_token_length),
              'dev_set': NetworkDatasetSAGEBert(to_absolute_path(cfg.dev_dataset_path),
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


# dataset = NetworkDatasetNode('../../data/neo_converted/nullptr_whole.pkl')
# g = dgl.from_networkx(dataset.graph)
#
# # Split edge set for training and testing
# u, v = g.edges()
#
# eids = np.arange(g.number_of_edges())
# eids = np.random.permutation(eids)
# test_size = int(len(eids) * 0.1)
# train_size = g.number_of_edges() - test_size
# test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
# train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
#
# # Find all negative edges and split them for training and testing
# neg_eids = np.arange(len(dataset.data['u']))
# test_neg_u, test_neg_v = dataset.u[neg_eids[:test_size]], dataset.v[neg_eids[:test_size]]
# train_neg_u, train_neg_v = dataset.u[neg_eids[test_size:]], dataset.v[neg_eids[test_size:]]
#
# train_g = dgl.remove_edges(g, eids[:test_size])
#
# train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
# train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
#
# test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
# test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

# DEVICE = torch.device('cuda:2')
#
# model = GraphSAGE(dataset.feature_dim, 128)
# model = model.to(DEVICE)
# You can replace DotPredictor with MLPPredictor.
# pred = MLPPredictor(16)
# pred = DotPredictor()
# pred = pred.to(DEVICE)

# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
# optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.001)

# ----------- 4. training -------------------------------- #
# all_logits = []
# graph_features = torch.tensor(dataset.node_features)
# graph_features = graph_features.to(DEVICE)
# g = g.to(DEVICE)
# train_pos_g = train_pos_g.to(DEVICE)
# train_neg_g = train_neg_g.to(DEVICE)
# train_g = train_g.to(DEVICE)
# for e in range(1000):
#     # forward
#     h = model(train_g, graph_features)
#     pos_score = pred(train_pos_g, h)
#     neg_score = pred(train_neg_g, h)
#     loss = compute_loss(pos_score, neg_score)
#
#     # backward
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if e % 10 == 0:
#         print('In epoch {}, loss: {}'.format(e, loss.detach().cpu()))


# test_pos_g = test_pos_g.to(DEVICE)
# test_neg_g = test_neg_g.to(DEVICE)
# with torch.no_grad():
#     pos_score = pred(test_pos_g, h)
#     neg_score = pred(test_neg_g, h)
#     print('AUC', compute_auc(pos_score, neg_score))


class MLPSolution(pl.LightningModule):
    def __init__(self, hydra_cfg: DictConfig):
        super().__init__()
        self.hydra_config = hydra_cfg
        self.model = MLPBert(bert_model_name=self.hydra_config.model.bert_model_name,
                             bert_num_feature=self.hydra_config.model.bert_num_feature)
        self.automatic_optimization = False
        self.save_hyperparameters(self.hydra_config)
        self.epoch_idx = 0
        self.accuracy_matrix = torchmetrics.Accuracy().to(self.device)

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
        label = torch_f.one_hot(y, num_classes=2)
        pred = self(u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts)

        loss = compute_train_loss(pred, label)

        self.manual_backward(loss)
        optimizer.step()
        return {'loss': loss}

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        loss.backward()

    def on_train_epoch_end(self) -> None:
        self.epoch_idx += 1
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                self.log('model_params' + name, param)

    def on_train_start(self) -> None:
        self.epoch_idx = 0

    def on_validation_epoch_start(self) -> None:
        self.accuracy_matrix.reset()

    def validation_step(self, batch, batch_idx):
        sample = batch
        u, v, u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts, y = sample
        label = torch_f.one_hot(y, num_classes=2)
        pred = self(u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts)
        loss = compute_train_loss(pred, label)

        answer: torch.Tensor = torch.softmax(pred, dim=-1)
        self.accuracy_matrix(answer, label)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['val_loss'].mean() for x in outputs]).mean()
        self.log('val_avg_loss', avg_loss)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hydra_config.train.lr, weight_decay=0.)


@hydra.main(config_path="conf", config_name="config")
def run(cfg):
    # >>>>>> Beg Configuration <<<<<<
    DEVICE = torch.device('cpu')
    print("Preparing tokenizer - Loading vocabs")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model_name)
    # <<<<<< End Configuration <<<<<<

    # >>>>>> Beg Init >>>>>>
    graphs = prepare_graph(cfg.dataset,
                           tokenizer,
                           DEVICE)
    train_loader = DataLoader(graphs['train_set'],
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=cfg.machine.num_workers,
                              persistent_workers=True)

    dev_loader = DataLoader(graphs['dev_set'],
                            batch_size=cfg.train.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=cfg.machine.num_workers,
                            persistent_workers=True)

    logger = pl_logger.TensorBoardLogger(save_dir=cfg.io.log_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints",
                                                       filename="{epoch}-{val_loss:.4f}",
                                                       monitor='val_loss',
                                                       save_last=True,
                                                       save_top_k=cfg.io.num_checkpoints,
                                                       mode='min',
                                                       save_weights_only=False,
                                                       every_n_epochs=1,
                                                       save_on_train_epoch_end=True)
    trainer = pl.Trainer(gpus=cfg.machine.gpus,
                         max_epochs=cfg.train.max_epochs,
                         callbacks=[checkpoint_callback,
                                    TQDMProgressBar(refresh_rate=10)],
                         checkpoint_callback=True,
                         # progress_bar_refresh_rate=10,
                         default_root_dir=cfg.io.checkpoint_dir,
                         logger=logger)
    solution = MLPSolution(cfg)
    trainer.fit(solution, train_loader, dev_loader)
    # <<<<<< Beg Init <<<<<<


if __name__ == '__main__':
    run()
