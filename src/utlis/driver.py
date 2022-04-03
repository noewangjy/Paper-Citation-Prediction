import pickle
from abc import ABC
from typing import Dict, Union

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm


class NetworkDatasetBase(Dataset, ABC):
    dataset_path: str
    graph = nx.Graph
    abstracts: np.ndarray
    authors: np.ndarray
    edges: np.ndarray
    u: np.ndarray
    v: np.ndarray
    y: np.ndarray
    u_tc: torch.Tensor
    v_tc: torch.Tensor
    y_tc: torch.Tensor
    edge_features: Union[None, np.ndarray]
    node_features: np.ndarray

    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path: str = dataset_path
        self.data = self.unpickle(self.dataset_path)
        self.unpack()
        assert len(self.authors) == len(self.abstracts) == len(self.graph.nodes()), ValueError(f"Data not aligned")

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

    @property
    def authors(self):
        return self.data['authors']

    @property
    def abstracts(self):
        return self.data['abstracts']

    @property
    def node_features(self):
        return self.data['node_features']

    @property
    def edge_features(self):
        return self.data['edge_features']

    def unpack(self):
        """
        This function prepare the dataset
        :return:
        """
        self.graph = nx.from_edgelist(self.data['origin_edges'])

    def __getitem__(self, item):
        """
        Implement custom function for different behavior
        :param item:
        :return:
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class NetworkDatasetEdge(NetworkDatasetBase):
    length: int
    dtype: torch.dtype
    edge_feature_dim: int = 5

    def __init__(self, dataset_path: str, dtype=torch.float32):
        super(NetworkDatasetBase, self).__init__(dataset_path)
        self.dtype = dtype
        self.length = len(self.edge_features)

    def __getitem__(self, item):
        return torch.tensor([self.u[item], self.v[item], self.edge_features[item]]).to(self.dtype)


class NetworkDatasetNode(NetworkDatasetBase):
    length: int
    dtype: torch.dtype

    def __init__(self, dataset_path: str, dtype=torch.float32):
        super(NetworkDatasetBase, self).__init__(dataset_path)
        self.dtype = dtype
        self.length = len(self.edge_features)

    def __getitem__(self, item):
        return torch.tensor([self.node_features[item]]).to(self.dtype)


class NetworkDatasetPassageMatching(NetworkDatasetBase):
    length: int

    def __init__(self,
                 dataset_path: str,
                 num_pos_neighbors: int = 5,
                 num_neg_neighbors: int = 5):
        # super(NetworkDatasetBase, self).__init__(dataset_path)
        NetworkDatasetBase.__init__(self, dataset_path)
        self.length = len(self.data['abstracts'])
        self.node_index = np.arange(self.length)
        self.num_pos_neighbors = num_pos_neighbors
        self.num_neg_neighbors = num_neg_neighbors

    def __getitem__(self, item):
        item_neighbors = list(self.graph[item].keys())
        np.random.shuffle(item_neighbors)
        pos_abstracts = ['' for _ in range(self.num_pos_neighbors)]
        pos_authors = ['' for _ in range(self.num_pos_neighbors)]
        for idx in range(min(self.num_pos_neighbors, len(item_neighbors))):
            pos_abstracts[idx] = self.abstracts[item_neighbors[idx]]
            pos_authors[idx] = ','.join(self.authors[item_neighbors[idx]])

        item_non_neighbors = []
        neg_abstracts = ['' for _ in range(self.num_neg_neighbors)]
        neg_authors = ['' for _ in range(self.num_neg_neighbors)]
        for idx in range(self.num_neg_neighbors):
            while True:
                node_idx = np.random.randint(0, self.length)
                if node_idx != item and node_idx not in item_neighbors and node_idx not in item_non_neighbors:
                    item_non_neighbors.append(node_idx)
                    neg_abstracts[idx] = self.abstracts[node_idx]
                    neg_authors[idx] = ','.join(self.authors[node_idx])
                    break

        return {'item': item,
                'pos_authors': pos_authors,
                'pos_abstracts': pos_abstracts,
                'neg_authors': neg_authors,
                'neg_abstracts': neg_abstracts}

    def __len__(self):
        return self.length


if __name__ == '__main__':
    # train_driver = NetworkDatasetBase('../../data/converted/nullptr_train.pkl')
    # dev_driver = NetworkDatasetBase('../../data/converted/nullptr_dev.pkl')
    # test_driver = NetworkDatasetBase('../../data/converted/nullptr_test.pkl')
    # whole_driver = NetworkDatasetBase('../../data/converted/nullptr_whole.pkl')

    # print(train_driver[1000:1100])
    # print(len(train_driver))
    train_driver = NetworkDatasetPassageMatching('../../data/neo_converted/nullptr_train.pkl')
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_driver, batch_size=32)

    print('finish')
