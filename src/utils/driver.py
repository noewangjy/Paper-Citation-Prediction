import pickle
from abc import ABC
from typing import Dict, Union, List, Tuple, Set
import os

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import dgl
import dgl.function as fn
import torch.nn.functional as torch_f


def compute_pagerank(graph: dgl.DGLGraph,
                     num_loops: int = 100,
                     DAMP: float = 0.85,
                     device: torch.device = torch.device('cpu')):
    # graph = dgl.from_networkx(graph)
    N = graph.number_of_nodes()
    graph.ndata['pv'] = torch.ones(N) / N
    degrees = graph.out_degrees(graph.nodes()).type(torch.float32)
    with tqdm(range(num_loops)) as pbar:
        pbar.set_description(f"Running PageRank")
        for k in range(num_loops):
            graph.ndata['pv'] = graph.ndata['pv'] / degrees
            graph.update_all(message_func=fn.copy_src(src='pv', out='m'),
                             reduce_func=fn.sum(msg='m', out='pv'))
            graph.ndata['pv'] = (1 - DAMP) / N + DAMP * graph.ndata['pv']
            pbar.update()
    return graph.ndata['pv']


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
        return len(self.u)


class NetworkDatasetEdge(NetworkDatasetBase):
    length: int
    dtype: torch.dtype
    edge_feature_dim: int = 5

    def __init__(self, dataset_path: str, dtype=torch.float32):
        NetworkDatasetBase.__init__(self, dataset_path)
        self.dtype = dtype
        self.length = len(self.u)
        self.shortest_path = None
        dgl_graph = dgl.graph((self.data['origin_edges'][:, 0], self.data['origin_edges'][:, 1]))
        self.page_rank = compute_pagerank(dgl_graph)
        self.features = self.generate_features()
        # self.katz_centrality = nx.katz_centrality(self.graph, max_iter=1000, tol=1e-5)

    @property
    def feature_dim(self) -> int:
        return -1

    def __getitem__(self, item):
        u = self.u[item]
        v = self.v[item]

        features = list()
        features.append(self.graph.degree[u] + self.graph.degree[v])
        features.append(abs(self.graph.degree[u] - self.graph.degree[v]))
        features.append(len(self.abstracts[u]) + len(self.abstracts[v]))
        features.append(abs(len(self.abstracts[u]) + len(self.abstracts[v])))
        features.append(len(set(self.abstracts[u].split()).intersection(set(self.abstracts[v].split()))))

        edge_features = {
            "nodes": torch.tensor([u, v]),
            "features": torch.tensor(features),
            "label": self.y[item]
        }

        return edge_features

    def generate_features(self):
        self.load_shortest_path()
        features = list()
        # uv = np.empty(shape=(len(self.u), 2))
        # uv[:, 0] = self.u
        # uv[:, 1] = self.v
        # jaccard_generator = nx.jaccard_coefficient(self.graph, uv)
        # adamic_adar_generator = nx.adamic_adar_index(self.graph, uv)

        for i in tqdm(range(self.length)):
            u = self.u[i]
            v = self.v[i]
            pr_u = float(self.page_rank[u])
            pr_v = float(self.page_rank[v])

            features.append([
                self.graph.degree[u] + self.graph.degree[v],  # -0.089
                abs(self.graph.degree[u] - self.graph.degree[v]),  # -0.034
                len(self.abstracts[u]) + len(self.abstracts[v]),  # -0.045
                abs(len(self.abstracts[u]) - len(self.abstracts[v])),  # -0.009
                len(set(self.abstracts[u].split()).intersection(set(self.abstracts[v].split()))),  # -0.073
                # len(",".join(self.authors[u])) + len(",".join(self.authors[v])), # useless
                # abs(len(self.authors[u]) - len(self.authors[v])),  # useless
                # len(self.abstracts[u].split()) + len(self.abstracts[v].split()),  # useless
                self.shortest_path[i],
                len(self._get_common_neighbors(u, v)),  # Common Neighbor
                self._get_jaccard_coefficient(u, v),    # Jaccard Coefficient, better than common neighbor
                self._get_adamic_adar(u, v),   # Adamic-adar coefficient
                self._get_resource_allocation(u, v),
                self._get_preferential_attachment(u, v),
                self._get_salton_cosine_similarity(u, v),

                pr_u + pr_v,
                abs(pr_u - pr_v),
                len(set(self.authors[u]).intersection(set(self.authors[v])))

            ])

        # norm = np.max(features, axis=10)
        # features[:, 10] /= norm
        return np.array(features)


    def _get_salton_cosine_similarity(self, u: int, v: int):
        common_neighbors = self._get_common_neighbors(u, v)
        return len(common_neighbors)/np.sqrt(len(self.graph[u])*len(self.graph[v]))

    def _get_resource_allocation(self, u: int, v: int):
        scores = 0
        neighbors = self._get_common_neighbors(u, v)
        for neighbor in neighbors:
            scores += len(self.graph[neighbor])
        return scores

    def _get_preferential_attachment(self, u: int, v: int):
        return len(self.graph[u])*len(self.graph[v])

    def _get_nodes_sp(self, u: int, v: int) -> int:
        if nx.has_path(self.graph, u, v):
            sp = nx.shortest_path_length(self.graph, u, v)
        else:
            sp = -1

        return sp

    def _get_jaccard_coefficient(self, u, v):
        u_neighbors = set(self.graph[u])
        v_neighbors = set(self.graph[v])
        return len(u_neighbors.intersection(v_neighbors)) / len(u_neighbors.union(v_neighbors))

    def _get_common_neighbors(self, u: int, v: int) -> Set:
        return set(self.graph[u]).intersection(set(self.graph[v]))

    def _get_adamic_adar(self, u: int, v: int) -> int:
        score = 0
        for node in self._get_common_neighbors(u, v):
            score += 1 / np.log(len(self.graph[node]))
        return score

    def get_shortest_path(self):
        shortest_paths: np.ndarray = np.zeros(self.length)
        for i in tqdm(range(self.length)):
            u = self.u[i]
            v = self.v[i]

            shortest_path: int = self._get_nodes_sp(u, v)
            if shortest_path == 1:
                for node in self.graph[u]:
                    if node == v:
                        continue
                    temp_sp = self._get_nodes_sp(node, v)
                    if temp_sp != -1:
                        shortest_path = temp_sp + 1
                        break

            shortest_paths[i] = shortest_path
        return shortest_paths

    def dump_shortest_path(self):
        with open(self.dataset_path.split("/")[-1].split(".")[0] + "_sp.pkl", "wb") as f:
            pickle.dump(self.get_shortest_path(), f)
        print("Shortest path successfully dumped")

    def load_shortest_path(self):
        with open(self.dataset_path.split("/")[-1].split(".")[0] + "_sp.pkl", "rb") as f:
            self.shortest_path = pickle.load(f)


class NetworkDatasetMLPBert(NetworkDatasetBase):
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
        res[:min(length, len(string_encoded))] = torch.tensor(string_encoded[:min(length, len(string_encoded))],
                                                              dtype=torch.int64)
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
        res[:min(length, len(string_encoded))] = torch.tensor(string_encoded[:min(length, len(string_encoded))],
                                                              dtype=torch.int64)
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

    def __init__(self, dataset_path: str):
        # super(NetworkDatasetBase, self).__init__(dataset_path)
        NetworkDatasetBase.__init__(self, dataset_path)
        self.length = len(self.u)

    def __getitem__(self, item):
        u = self.u[item]
        v = self.v[item]
        y = self.y[item]
        query = {
            'authors': ','.join(self.authors[u]),
            'abstract': self.abstracts[u]
        }
        context = {
            'authors': ','.join(self.authors[v]),
            'abstract': self.abstracts[v]
        }

        return {
            'query': query,
            'context': context,
            'label': y
        }

    def __len__(self):
        return self.length


class NetworkDatasetEmbeddingClassification(NetworkDatasetBase):
    def __init__(self, dataset_path: str):
        # super(NetworkDatasetBase, self).__init__(dataset_path)
        NetworkDatasetBase.__init__(self, dataset_path)
        self.length = len(self.u)

    def __getitem__(self, item):
        return {
            'u': torch.tensor(self.u[item], dtype=torch.long),
            'v': torch.tensor(self.v[item], dtype=torch.long),
            'y': self.y[item]
        }

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
    train_driver = NetworkDatasetMLPBert('../../data/neo_converted/nullptr_no_feature_train.pkl', tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_driver = NetworkDatasetGraphSAGEBert('../../data/neo_converted/nullptr_no_feature_train.pkl',
                                               '../../data/abstract_features_v1/features.pkl')
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_driver, batch_size=32)
    train_iterator = iter(train_loader)
    sample = next(train_iterator)
    print('finish')
