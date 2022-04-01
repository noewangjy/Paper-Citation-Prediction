import os
import pickle
from typing import Dict, List

import networkx as nx
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset


class AbstracDriver(Dataset):
    def __init__(self):
        super().__init__()

    def get_cached_result(self, record_name: str) -> bool:
        cache_name: str = os.path.join(os.path.splitext(record_name)[0] + '.pkl')
        if os.path.exists(cache_name):
            with open(cache_name, "rb") as f:
                pickle.load(f)
            return True
        else:
            return False


class DataDrver(AbstracDriver):
    __record_names__: Dict[str, str] = {
        "abstracts": "abstracts.txt",
        "authors": "authors.txt",
        "edges": "edgelist.txt",
        "evaluate_cases": "test.txt"
    }
    dataset_path: str
    abstracts: np.ndarray
    authors: np.ndarray
    graph = nx.Graph
    edges = np.ndarray

    @classmethod
    def read_abstracts(cls, abstracts_path: str) -> np.ndarray:
        def split_fn(line: str) -> List[str]:
            index, abstract = line.split('|--|')
            return abstract

        with open(abstracts_path, 'r') as f:
            result = map(split_fn, f.readlines())

        return np.array(list(result), dtype=object)

    @classmethod
    def read_author(cls, authors_path: str) -> np.ndarray:
        def split_fn(line: str) -> List[str]:
            _, authors = line.split('|--|')
            return authors.split(',')

        with open(authors_path, 'r') as f:
            result = map(split_fn, f.readlines())

        return np.array(list(result), dtype=object)

    @classmethod
    def read_graph(cls, edges_path: str, nodetype=int):
        return nx.read_edgelist(edges_path, delimiter=',', create_using=nx.Graph(), nodetype=nodetype)

    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path: str = dataset_path
        self.abstracts = self.read_abstracts(os.path.join(dataset_path, self.__record_names__['abstracts']))
        self.authors = self.read_author(os.path.join(dataset_path, self.__record_names__['authors']))
        self.graph = self.read_graph(os.path.join(dataset_path, self.__record_names__['edges']))
        self.edges = np.array(self.graph.edges())

        assert len(self.authors) == len(self.abstracts) == len(self.graph), ValueError(f"Data not aligned")

        self.warm_up()
        pass

    def warm_up(self):
        """
        This function prepare the train/test/eval dataset
        :return:
        """
        pos_u, pos_v = self.edges[:, 0], self.edges[:, 1]
        self.number_of_edges: int = self.graph.number_of_edges()

        pos_edge_ids = np.arange(self.number_of_edges)
        pos_edge_ids = np.random.permutation(pos_edge_ids)

        test_size = eval_size = int(len(pos_edge_ids) * 0.1)
        train_size = self.number_of_edges - (test_size + eval_size)

        train_pos_u, train_pos_v = pos_u[pos_edge_ids[:train_size]], pos_v[pos_edge_ids[:train_size]]
        test_pos_u, test_pos_v = pos_u[pos_edge_ids[train_size:train_size + test_size]], pos_v[pos_edge_ids[train_size:train_size + test_size]]
        eval_pos_u, eval_pos_v = pos_u[pos_edge_ids[train_size + test_size:]], pos_v[pos_edge_ids[train_size + test_size:]]

        # 采样所有负样例并划分为训练集和测试集中。
        adj = sp.coo_matrix((np.ones(len(pos_u)), (pos_u, pos_v)))
        adj_neg = 1 - adj.todense() - np.eye(self.number_of_edges)
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_edge_ids = np.random.choice(len(neg_u), self.number_of_edges)
        train_pos_u, train_pos_v = neg_u[neg_edge_ids[:train_size]], neg_v[neg_edge_ids[:train_size]]
        test_pos_u, test_pos_v = neg_u[neg_edge_ids[train_size:train_size + test_size]], neg_v[neg_edge_ids[train_size:train_size + test_size]]
        eval_pos_u, eval_pos_v = neg_u[neg_edge_ids[train_size + test_size:]], neg_v[neg_edge_ids[train_size + test_size:]]

        # TODO: Finish this part
        self.length = len(self.authors)  # TODO: This setting is wrong
        pass

    def __len__(self):
        return self.length


if __name__ == '__main__':
    driver = DataDrver('../../data')
    print('finish')
