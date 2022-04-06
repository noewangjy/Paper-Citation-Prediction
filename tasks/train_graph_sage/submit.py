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
import sys

from src.models import GraphSAGEBundled
from src.utils.driver import NetworkDatasetGraphSAGEBert
from src.utils.submmision import generate_submission


@hydra.main(config_path='conf', config_name='submit')
def run(cfg):
    # >>>>>> Beg Configuration >>>>>>
    DEVICE = torch.device('cuda:3')
    # DEVICE = torch.device('cpu') if len(cfg.machine.gpus) <= 0 else torch.device(f'cuda:{cfg.machine.gpus[0]}')
    test_set = NetworkDatasetGraphSAGEBert(to_absolute_path(cfg.test_dataset_path),
                                           to_absolute_path(cfg.features_path))
    train_set = NetworkDatasetGraphSAGEBert(to_absolute_path(cfg.test_dataset_path),
                                            to_absolute_path(cfg.features_path))
    features = test_set.node_features.to(DEVICE)
    num_features = test_set.data['node_features'].shape[1]
    num_nodes = test_set.graph.number_of_nodes()

    test_graph = dgl.graph((test_set.u, test_set.v), num_nodes=num_nodes).to(DEVICE)
    train_graph = dgl.graph((train_set.u, train_set.v), num_nodes=num_nodes).to(DEVICE)

    model = GraphSAGEBundled(num_features,
                             cfg.model.hidden_dims, cfg.model.predictor)
    model.load_state_dict({".".join(k.split(".")[1:]): v.to(DEVICE) for k, v in torch.load(to_absolute_path(cfg.state_dict_path))['state_dict'].items()})
    model.to(DEVICE)
    # <<<<<< End Configuration <<<<<<

    h = model(train_graph, features)
    scores = model.predictor(test_graph, h)
    scores = torch.sigmoid(scores)
    scores_np = scores.detach().cpu().numpy()
    generate_submission('.', scores_np)


if __name__ == '__main__':
    run()
