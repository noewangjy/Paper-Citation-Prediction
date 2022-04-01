import pickle
from abc import ABC
from typing import Dict

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset


class NetworkDataset(Dataset, ABC):
    dataset_path: str
    graph = nx.Graph
    abstracts: np.ndarray
    authors: np.ndarray
    edges: np.ndarray
    length: int
    u: np.ndarray
    v: np.ndarray
    y: np.ndarray
    u_tc: torch.Tensor
    v_tc: torch.Tensor
    y_tc: torch.Tensor

    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path: str = dataset_path
        self.data = self.unpickle(self.dataset_path)
        self.unpack()
        assert len(self.authors) == len(self.abstracts) == len(self.graph.nodes()), ValueError(f"Data not aligned")

        pass

    @classmethod
    def unpickle(cls, dataset_path: str) -> Dict[str, np.ndarray]:
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)

    def unpack(self):
        """
        This function prepare the train/test/eval dataset
        :return:
        """
        self.u = np.concatenate([self.data['pos_u'], self.data['neg_u']])
        self.v = np.concatenate([self.data['pos_v'], self.data['neg_v']])
        self.y = np.concatenate([np.ones(shape=len(self.data['pos_u']), dtype=int), np.zeros(shape=len(self.data['pos_u']), dtype=int)])

        self.u_tc = torch.tensor(self.u, dtype=torch.float32)
        self.v_tc = torch.tensor(self.v, dtype=torch.float32)
        self.y_tc = torch.tensor(self.y, dtype=torch.float32)

        self.graph = nx.from_edgelist(self.data['origin_edges'])
        self.authors = self.data['authors']
        self.abstracts = self.data['abstracts']
        self.length = self.u.shape[0]

    def get_item_baseline(self, item):
        u, v, y = self.u[item], self.v[item], self.y[item]
        u_degree = self.graph.degree(u)
        v_degree = self.graph.degree(v)
        u_abstract = self.abstracts[u]
        v_abstract = self.abstracts[v]

        return v_degree + u_degree, \
               abs(v_degree - u_degree), \
               len(u_abstract) + len(v_abstract), \
               abs(len(u_abstract) + len(v_abstract)), \
               len(set(u_abstract.split()).intersection(set(v_abstract.split()))), \
               y

    def __getitem__(self, item):
        """
        Implement custom function for different behavior
        :param item:
        :return:
        """
        return self.get_item_baseline(item)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    driver = NetworkDataset('../../data/converted/nullptr_train.pkl')
    print(driver[1000])
    print(len(driver))
    print('finish')
