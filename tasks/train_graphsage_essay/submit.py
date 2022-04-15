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
import pickle

from src.models import GraphSAGEBundled
from src.utils.driver import NetworkDatasetGraphSAGEBert
from src.utils.submmision import generate_submission

from train import prepare_graph


@hydra.main(config_path='conf', config_name='submit')
def run(cfg):
    # >>>>>> Beg Configuration >>>>>>
    DEVICE = torch.device('cuda:3')
    # DEVICE = torch.device('cpu') if len(cfg.machine.gpus) <= 0 else torch.device(f'cuda:{cfg.machine.gpus[0]}')
    graphs = prepare_graph(cfg, device=DEVICE)

    model = GraphSAGEBundled(graphs['num_features'],
                             cfg.model.hidden_dims, cfg.model.predictor)
    model.load_state_dict({".".join(k.split(".")[1:]): v.to(DEVICE) for k, v in torch.load(to_absolute_path(cfg.state_dict_path))['state_dict'].items()})
    model.to(DEVICE)
    # <<<<<< End Configuration <<<<<<

    train_graph = graphs['train_graph']
    features = graphs['features']
    test_graph = graphs['test_graph']
    h = model(train_graph, features)
    scores = model.predictor(test_graph, h)
    scores_sigmoid = torch.sigmoid(scores)
    scores_np = scores.detach().cpu().numpy()
    scores_sigmoid_np = scores_sigmoid.detach().cpu().numpy()
    generate_submission('.', scores_np, 'no_sigmoid')
    generate_submission('.', scores_sigmoid_np, 'sigmoid')


if __name__ == '__main__':
    run()
