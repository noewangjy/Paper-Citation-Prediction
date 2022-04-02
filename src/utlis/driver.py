import pickle
from abc import ABC
from typing import Dict, Union

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm


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
    cache: Union[None, np.ndarray]

    def __init__(self, dataset_path: str, build_cache: bool = True):
        super().__init__()
        self.dataset_path: str = dataset_path
        self.data = self.unpickle(self.dataset_path)
        self.unpack()
        assert len(self.authors) == len(self.abstracts) == len(self.graph.nodes()), ValueError(f"Data not aligned")
        self.cache = self.data['cache'] if 'cache' in self.data.keys() else None
        if build_cache and self.cache is None:
            self.build_cache()

    @classmethod
    def unpickle(cls, dataset_path: str) -> Dict[str, np.ndarray]:
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)

    @property
    def u(self):
        return self.data['u']

    @property
    def v(self):
        return self.data['v']

    @property
    def y(self):
        return self.data['y']

    def unpack(self):
        """
        This function prepare the train/test/eval dataset
        :return:
        """
        self.u_tc = torch.tensor(self.u, dtype=torch.float32)
        self.v_tc = torch.tensor(self.v, dtype=torch.float32)
        self.y_tc = torch.tensor(self.y, dtype=torch.float32)

        self.graph = nx.from_edgelist(self.data['origin_edges'])
        self.authors = self.data['authors']
        self.abstracts = self.data['abstracts']
        self.length = self.u.shape[0]

    def build_cache(self):
        cache = np.zeros(shape=(self.length, self.feature_dim + 1), dtype=np.float32)
        with tqdm.tqdm(range(self.length)) as pbar:
            for idx in range(self.length):
                cache[idx] = self[idx]
                pbar.update()
        self.cache = cache
        self.data['cache'] = cache
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self.data, f)

    feature_dim: int = 5  # __getitem__ output dim = feature_dim + 1

    def get_item_baseline(self, item):
        u, v, y = self.u[item], self.v[item], self.y[item]
        u_degree = int(self.graph.degree(u))
        v_degree = int(self.graph.degree(v))
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
        if self.cache is not None:
            return self.cache[item]
        else:
            return self.get_item_baseline(item)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    train_driver = NetworkDataset('../../data/converted/nullptr_train.pkl')
    dev_driver = NetworkDataset('../../data/converted/nullptr_dev.pkl')
    test_driver = NetworkDataset('../../data/converted/nullptr_test.pkl')
    whole_driver = NetworkDataset('../../data/converted/nullptr_whole.pkl')

    print(train_driver[1000:1100])
    print(len(train_driver))
    print('finish')
