import json
import os
import pickle
import sys
from typing import Dict, List, Tuple, Any, Union

import networkx as nx
import numpy as np
import torch
import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import feature_fns
import logging


def read_abstracts(abstracts_path: str) -> np.ndarray:
    def split_fn(line: str) -> List[str]:
        line = line.replace('\n', '')
        index, abstract = line.split('|--|')
        return abstract

    with open(abstracts_path, 'r') as f:
        result = map(split_fn, f.readlines())

    return np.array(list(result), dtype=object)


def read_authors(authors_path: str) -> np.ndarray:
    def split_fn(line: str) -> List[str]:
        line = line.replace('\n', '')
        _, authors = line.split('|--|')
        return authors.split(',')

    with open(authors_path, 'r') as f:
        result = map(split_fn, f.readlines())

    return np.array(list(result), dtype=object)


def read_graph(edges_path: str, nodetype=int) -> nx.Graph:
    return nx.read_edgelist(edges_path, delimiter=',', create_using=nx.Graph(), nodetype=nodetype)


def read_dataset_context(cfg: DictConfig):
    print(f"Reading dataset")
    context: Dict[str, Any] = {
        'graph': read_graph(os.path.join(cfg.read_path, cfg.record_names.edges)),
        'abstracts': read_abstracts(os.path.join(cfg.read_path, cfg.record_names.abstracts)),
        'authors': read_authors(os.path.join(cfg.read_path, cfg.record_names.authors)),
    }
    assert len(context['authors']) == len(context['abstracts']) == len(context['graph']), ValueError(f'Data not aligned')
    number_of_edges: int = context['graph'].number_of_edges()
    pos_edge_ids = np.arange(number_of_edges)
    neg_edge_ids = np.arange(number_of_edges)
    context['ids'] = {
        'pos': np.random.permutation(pos_edge_ids),
        'neg': np.random.permutation(neg_edge_ids)
    }
    context['size'] = {
        'dev': int(number_of_edges * cfg.dev_ratio),
        'train': number_of_edges - int(number_of_edges * cfg.dev_ratio),
        'whole': number_of_edges
    }

    return context


def generate_test_dataset(cfg: DictConfig) -> Dict[str, np.ndarray]:
    def read_evaluate_cases(test_path: str) -> np.ndarray:
        node_pairs = list()
        with open(test_path, 'r') as f:
            for line in f:
                t = line.split(',')
                node_pairs.append((int(t[0]), int(t[1])))

        return np.array(node_pairs)

    evaluate_cases: np.ndarray = read_evaluate_cases(os.path.join(cfg.read_path, cfg.record_names.evaluate_cases))
    test_set = {'u': evaluate_cases[:, 0],
                'v': evaluate_cases[:, 1],
                'y': - np.ones_like(evaluate_cases[:, 0])}

    return test_set


def save_datasets(cfg: DictConfig, datasets: Dict[str, Dict[str, Any]]):
    # Saving
    dataset_type_name = "_".join(cfg.target_features.keys())
    print(dataset_type_name)
    print(f"Saving to {cfg.output_path}")
    for name in datasets.keys():
        with open(os.path.join(cfg.output_path, f'{cfg.prefix}_{name}.pkl'), 'wb') as f:
            pickle.dump(datasets[name], f)
    with open(os.path.join(cfg.output_path, f'info.txt'), 'w') as f:
        f.write(dataset_type_name)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg.read_path = to_absolute_path(cfg.read_path)
    cfg.output_path = to_absolute_path(cfg.output_path)
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    logger = logging.getLogger(__name__)

    context = read_dataset_context(cfg)

    datasets: Dict[str, Dict[str, Any]] = {
        'train': {},
        'dev': {},
        'whole': {},
    }

    for feature_name in cfg.target_features.keys():
        datasets = getattr(feature_fns, f'feature_fn_{feature_name}')(cfg, context, datasets, logger)

    datasets['test'] = generate_test_dataset(cfg)
    save_datasets(cfg, datasets)


if __name__ == '__main__':
    main()
