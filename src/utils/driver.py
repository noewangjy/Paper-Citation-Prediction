import pickle
from abc import ABC
from typing import Dict, Union, List
import os

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as torch_f


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
    length: int = 0

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
    def pos_u(self):
        return self.data['pos_u']

    @property
    def neg_u(self):
        return self.data['neg_u']

    @property
    def v(self):
        return self.data['v']

    @property
    def pos_v(self):
        return self.data['pos_v']

    @property
    def neg_v(self):
        return self.data['neg_v']

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
        return self.u[item], self.v[item]

    def __len__(self):
        return self.length


class NetworkDatasetEdge(NetworkDatasetBase):
    length: int
    dtype: torch.dtype
    edge_feature_dim: int = 5

    def __init__(self, dataset_path: str, dtype=torch.float32):
        NetworkDatasetBase.__init__(self, dataset_path)
        self.dtype = dtype
        self.length = len(self.edge_features)

    @property
    def feature_dim(self) -> int:
        return self.edge_features.shape[1] + 3

    def __getitem__(self, item):
        return torch.cat([torch.tensor([self.u[item],
                                        self.v[item]]),
                          torch.tensor(self.edge_features[item]),
                          torch.tensor([self.y[item]])]).to(self.dtype)


class NetworkDatasetMLPBert(NetworkDatasetBase):
    length: int

    def __init__(self,
                 dataset_path: str,
                 tokenizer: AutoTokenizer,
                 author_token_length: int = 128,
                 abstract_token_length: int = 512,
                 pos_edges_only: bool = False):
        NetworkDatasetBase.__init__(self, dataset_path)
        self.tokenizer = tokenizer
        self.pos_edges_only = pos_edges_only
        if self.pos_edges_only:
            self.length = len(self.data['pos_u'])
        else:
            self.length = len(self.data['u'])
        self.author_token_length = author_token_length
        self.abstract_token_length = abstract_token_length

    @property
    def feature_dim(self) -> int:
        return self.edge_features.shape[1] + 3

    def convert_to_token(self, string: str, length: int) -> torch.Tensor:
        res = torch.zeros(size=(length,), dtype=torch.int64)
        string_encoded = self.tokenizer.encode(string, truncation=True, max_length=length)
        res[:min(length, len(string_encoded))] = torch.tensor(string_encoded[:min(length, len(string_encoded))], dtype=torch.int64)
        return res

    def __getitem__(self, item):
        if self.pos_edges_only:
            u, v, y = self.pos_u[item], self.pos_v[item], 1
        else:
            u, v, y = self.u[item], self.v[item], self.y[item]
        u_authors = self.convert_to_token(','.join(self.authors[u]), self.author_token_length)
        v_authors = self.convert_to_token(','.join(self.authors[v]), self.author_token_length)

        u_abstract = self.convert_to_token(self.abstracts[u], self.abstract_token_length)
        v_abstract = self.convert_to_token(self.abstracts[v], self.abstract_token_length)

        return torch.tensor([u], dtype=torch.int64), \
               torch.tensor([v], dtype=torch.int64), \
               torch.tensor([self.graph.degree(u)], dtype=torch.int64), \
               torch.tensor([self.graph.degree(v)], dtype=torch.int64), \
               torch.tensor([abs(self.graph.degree(u) - self.graph.degree(v))], dtype=torch.int64), \
               u_authors, \
               v_authors, \
               u_abstract, \
               v_abstract, \
               torch.tensor([y], dtype=torch.int64)

    @property
    def keys(self):
        return {
            'u': 0,
            'v': 1,
            'u_deg': 2,
            'v_deg': 3,
            'uv_deg_diff': 4,
            'u_authors': 5,
            'v_authors': 6,
            'u_abstracts': 7,
            'v_abstracts': 8,
            'y': 9
        }


class NetworkDatasetSAGEBert(NetworkDatasetBase):
    length: int

    def __init__(self,
                 dataset_path: str,
                 tokenizer: AutoTokenizer,
                 author_token_length: int = 128,
                 abstract_token_length: int = 512):
        NetworkDatasetBase.__init__(self, dataset_path)
        self.tokenizer = tokenizer
        self.length = len(self.data['u'])
        self.author_token_length = author_token_length
        self.abstract_token_length = abstract_token_length

    @property
    def feature_dim(self) -> int:
        return self.edge_features.shape[1] + 3

    def convert_to_token(self, string: str, length: int) -> torch.Tensor:
        res = torch.zeros(size=(length,), dtype=torch.int64)
        string_encoded = self.tokenizer.encode(string, truncation=True, max_length=length)
        res[:min(length, len(string_encoded))] = torch.tensor(string_encoded[:min(length, len(string_encoded))], dtype=torch.int64)
        return res

    def __getitem__(self, item):
        u, v, y = self.u[item], self.v[item], self.y[item]
        u_authors = self.convert_to_token(','.join(self.authors[u]), self.author_token_length)
        v_authors = self.convert_to_token(','.join(self.authors[v]), self.author_token_length)

        u_abstract = self.convert_to_token(self.abstracts[u], self.abstract_token_length)
        v_abstract = self.convert_to_token(self.abstracts[v], self.abstract_token_length)

        return torch.tensor([u], dtype=torch.int64), \
               torch.tensor([v], dtype=torch.int64), \
               torch.tensor([self.graph.degree(u)], dtype=torch.int64), \
               torch.tensor([self.graph.degree(v)], dtype=torch.int64), \
               torch.tensor([abs(self.graph.degree(u) - self.graph.degree(v))], dtype=torch.int64), \
               u_authors, \
               v_authors, \
               u_abstract, \
               v_abstract, \
               y

    @property
    def keys(self):
        return {
            'u': 0,
            'v': 1,
            'u_deg': 2,
            'v_deg': 3,
            'uv_deg_diff': 4,
            'u_authors': 5,
            'v_authors': 6,
            'u_abstracts': 7,
            'v_abstracts': 8,
            'y': 9
        }


class NetworkDatasetNode(NetworkDatasetBase):
    length: int
    dtype: torch.dtype

    def __init__(self, dataset_path: str, dtype=torch.float32):
        NetworkDatasetBase.__init__(self, dataset_path)
        self.dtype = dtype
        self.length = len(self.edge_features)

    @property
    def feature_dim(self) -> int:
        return self.node_features.shape[1]

    def __getitem__(self, item):
        return torch.tensor([self.node_features[item]]).to(self.dtype)


class NetworkDatasetGraphSAGEBert(NetworkDatasetBase):
    length: int
    dtype: torch.dtype

    def __init__(self,
                 dataset_path: str,
                 feature_path: str,
                 dtype: torch.dtype = torch.float32):
        NetworkDatasetBase.__init__(self, dataset_path)
        self.dtype = dtype
        self.data['node_features'] = self.unpack_features(pickle.load(open(feature_path, 'rb')), self.dtype)
        self.length = len(self.node_features)

    @staticmethod
    def unpack_features(features: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        features_sum = np.sum(features, axis=1, keepdims=True)
        return torch.tensor(features.astype(int) / features_sum, dtype=dtype)

    @property
    def feature_dim(self) -> int:
        return self.node_features.shape[1]

    def __getitem__(self, item):
        return self.node_features[item]


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

        pos_passage_idx_list: List[int] = [-1 for _ in range(self.num_pos_neighbors)]
        pos_abstracts = ['' for _ in range(self.num_pos_neighbors)]
        pos_authors = ['' for _ in range(self.num_pos_neighbors)]
        for idx in range(min(self.num_pos_neighbors, len(item_neighbors))):
            pos_passage_idx_list[idx] = item_neighbors[idx]
            pos_abstracts[idx] = self.abstracts[item_neighbors[idx]]
            pos_authors[idx] = ','.join(self.authors[item_neighbors[idx]])

        neg_passage_idx_list: List[int] = []
        neg_abstracts = ['' for _ in range(self.num_neg_neighbors)]
        neg_authors = ['' for _ in range(self.num_neg_neighbors)]
        for idx in range(self.num_neg_neighbors):
            while True:
                node_idx = np.random.randint(0, self.length)
                if node_idx != item and node_idx not in item_neighbors and node_idx not in neg_passage_idx_list:
                    neg_passage_idx_list.append(node_idx)
                    neg_abstracts[idx] = self.abstracts[node_idx]
                    neg_authors[idx] = ','.join(self.authors[node_idx])
                    break

        return {'item': item,
                'authors': ','.join(self.authors[item]),
                'abstracts': self.abstracts[item],
                'pos_passages': torch.tensor(pos_passage_idx_list),
                'neg_passages': torch.tensor(neg_passage_idx_list),
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
    # train_driver = NetworkDatasetPassageMatching('../../data/neo_converted/nullptr_train.pkl')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_driver = NetworkDatasetMLPBert('../../data/neo_converted/nullptr_no_feature_train.pkl', tokenizer, pos_edges_only=True)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # train_driver = NetworkDatasetGraphSAGEBert('../../data/neo_converted/nullptr_no_feature_train.pkl', '../../data/abstract_features_v1/features.pkl')
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_driver, batch_size=32)
    train_iterator = iter(train_loader)
    sample = next(train_iterator)
    print('finish')
