import dgl
import numpy as np
import networkx as nx
from typing import List, Tuple
import os
import pickle

import torch


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


def dump_features(output_dir: str,
                  X_whole,
                  Y_whole,
                  X_train,
                  Y_train,
                  X_dev,
                  Y_dev,
                  X_test,
                  tag: str = ""):
    print(f"Dumping features to {output_dir} with tag={tag}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    res = {
        'X_whole': X_whole,
        'Y_whole': Y_whole,
        'X_train': X_train,
        'Y_train': Y_train,
        'X_dev': X_dev,
        'Y_dev': Y_dev,
        'X_test': X_test,
    }
    with open(os.path.join(output_dir, tag + f'{"_".join([tag, "features"])}.pkl'), 'wb') as f:
        pickle.dump(res, f)


def split_graph(pos_edges: torch.Tensor,
                ratio: float = 0.2,
                seed: int = 0,
                device: torch.device = torch.device('cpu')):
    np.random.seed(seed)
    if not isinstance(pos_edges, torch.Tensor):
        pos_edges = torch.tensor(pos_edges)

    num_pos_edges: int = len(pos_edges)
    train_size: int = int(num_pos_edges * (1 - ratio))
    indices: np.ndarray = np.random.permutation(np.arange(0, num_pos_edges))
    train_indices, dev_indices = indices[:train_size], indices[train_size:]

    whole_pos_graph: dgl.DGLGraph = dgl.graph((pos_edges[:, 0], pos_edges[:, 1]))
    num_nodes: int = whole_pos_graph.num_nodes()
    neg_edges: Tuple[torch.Tensor] = dgl.sampling.global_uniform_negative_sampling(whole_pos_graph, num_pos_edges)
    neg_edges: torch.Tensor = torch.cat([neg_edges[0].unsqueeze(1), neg_edges[1].unsqueeze(1)], dim=1)

    train_edges = torch.cat([pos_edges[train_indices], neg_edges[train_indices]], dim=0)
    train_labels = torch.cat([torch.ones(size=(len(train_indices),)), torch.zeros(size=(len(train_indices),))])
    dev_edges = torch.cat([pos_edges[dev_indices], neg_edges[dev_indices]], dim=0)
    dev_labels = torch.cat([torch.ones(size=(len(dev_indices),)), torch.zeros(size=(len(dev_indices),))])

    return {
        'num_nodes': num_nodes,
        'whole_graph': whole_pos_graph.to(device),
        'whole_pos_graph': dgl.graph((pos_edges[:, 0], pos_edges[:, 1]), num_nodes=num_nodes).to(device),
        'whole_neg_graph': dgl.graph((neg_edges[:, 0], neg_edges[:, 1]), num_nodes=num_nodes).to(device),
        'train_graph': dgl.graph((train_edges[:, 0], train_edges[:, 1]), num_nodes=num_nodes).to(device),
        'train_pos_graph': dgl.graph((pos_edges[train_indices][:, 0], pos_edges[train_indices][:, 1]), num_nodes=num_nodes).to(device),
        'train_neg_graph': dgl.graph((neg_edges[train_indices][:, 0], neg_edges[train_indices][:, 1]), num_nodes=num_nodes).to(device),
        'dev_graph': dgl.graph((dev_edges[:, 0], dev_edges[:, 1]), num_nodes=num_nodes).to(device),
        'dev_pos_graph': dgl.graph((pos_edges[dev_indices][:, 0], pos_edges[dev_indices][:, 1]), num_nodes=num_nodes).to(device),
        'dev_neg_graph': dgl.graph((neg_edges[dev_indices][:, 0], neg_edges[dev_indices][:, 1]), num_nodes=num_nodes).to(device),
        'train_edges': train_edges.to(device),
        'train_labels': train_labels.to(device),
        'dev_edges': dev_edges.to(device),
        'dev_labels': dev_labels.to(device),
    }


def check_md5(path, display: bool = True):
    res = os.popen(f'md5sum {path}').readlines()[0].split(' ')[0]
    if display:
        print(f"MD5SUM of {path}: {res}")
    return res
