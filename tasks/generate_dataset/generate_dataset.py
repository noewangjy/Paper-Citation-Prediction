import os
import pickle
import sys
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import tqdm

from src.utlis import NetworkDataset


def read_abstracts(abstracts_path: str) -> np.ndarray:
    def split_fn(line: str) -> List[str]:
        index, abstract = line.split('|--|')
        return abstract

    with open(abstracts_path, 'r') as f:
        result = map(split_fn, f.readlines())

    return np.array(list(result), dtype=object)


def read_authors(authors_path: str) -> np.ndarray:
    def split_fn(line: str) -> List[str]:
        _, authors = line.split('|--|')
        return authors.split(',')

    with open(authors_path, 'r') as f:
        result = map(split_fn, f.readlines())

    return np.array(list(result), dtype=object)


def read_graph(edges_path: str, nodetype=int) -> nx.Graph:
    return nx.read_edgelist(edges_path, delimiter=',', create_using=nx.Graph(), nodetype=nodetype)


def read_evaluate_cases(test_path: str) -> np.ndarray:
    node_pairs = list()
    with open(test_path, 'r') as f:
        for line in f:
            t = line.split(',')
            node_pairs.append((int(t[0]), int(t[1])))

    return np.array(node_pairs)


# @numba.jit(nopython=True)
def select_neg_edges(G: nx.Graph,
                     num_neg_edges: int) -> Tuple[np.ndarray, np.ndarray]:
    num_nodes = G.number_of_nodes()
    neg_u, neg_v = np.zeros(shape=(num_neg_edges,), dtype=int), np.zeros(shape=(num_neg_edges,), dtype=int)
    with tqdm.tqdm(range(num_neg_edges)) as pbar:
        for idx in range(num_neg_edges):
            while True:
                i, j = np.random.randint(0, num_nodes), np.random.randint(0, num_nodes)
                if not j in G[i].keys():
                    neg_u[idx], neg_v[idx] = i, j
                    pbar.update()
                    break

    return neg_u, neg_v


def create_features(G: nx.Graph,
                    num_edges: int):
    with tqdm.tqdm(range(num_edges)):
        pass


if __name__ == '__main__':
    # >>>>>> Beg modify parameters >>>>>>
    RECORD_NAMES: Dict[str, str] = {
        "abstracts": "abstracts.txt",
        "authors": "authors.txt",
        "edges": "edgelist.txt",
        "evaluate_cases": "test.txt"
    }
    DEV_RATIO: float = 0.2
    DATASET_PATH = input("Input the path of dataset") if len(sys.argv) < 2 else sys.argv[1]
    OUTPUT_PATH = input("Input the path of output") if len(sys.argv) < 3 else sys.argv[2]
    SEED = 0
    PREFIX: str = 'nullptr'
    # <<<<<< End modify parameters <<<<<<
    np.random.seed(SEED)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    whole_set: Dict[str, np.ndarray] = {}
    train_set: Dict[str, np.ndarray] = {}
    dev_set: Dict[str, np.ndarray] = {}
    test_set: Dict[str, np.ndarray] = {}

    # Assign common information
    dataset_path: str = DATASET_PATH
    graph = nx.Graph = read_graph(os.path.join(DATASET_PATH, RECORD_NAMES['edges']))
    abstracts: np.ndarray = read_abstracts(os.path.join(DATASET_PATH, RECORD_NAMES['abstracts']))
    authors: np.ndarray = read_authors(os.path.join(DATASET_PATH, RECORD_NAMES['authors']))
    evaluate_cases: np.ndarray = read_evaluate_cases(os.path.join(DATASET_PATH, RECORD_NAMES['evaluate_cases']))
    edges: np.ndarray = np.array(graph.edges())
    train_set['abstracts'], dev_set['abstracts'], test_set['abstracts'], whole_set['abstracts'] = abstracts, abstracts, abstracts, abstracts
    train_set['authors'], dev_set['authors'], test_set['authors'], whole_set['authors'] = authors, authors, authors, authors
    train_set['origin_edges'], dev_set['origin_edges'], test_set['origin_edges'], whole_set['origin_edges'] = edges, edges, edges, edges

    assert len(authors) == len(abstracts) == len(graph), ValueError(f"Data not aligned")

    number_of_edges: int = graph.number_of_edges()
    pos_edge_ids = np.arange(number_of_edges)
    pos_edge_ids = np.random.permutation(pos_edge_ids)
    neg_edge_ids = np.arange(number_of_edges)
    neg_edge_ids = np.random.permutation(neg_edge_ids)

    # Calculate test size and train size
    dev_size = int(len(pos_edge_ids) * DEV_RATIO)
    train_size = number_of_edges - dev_size

    # Sample positive cases
    print(f"Sample positive cases")
    pos_u, pos_v = edges[:, 0], edges[:, 1]
    train_set['pos_u'], train_set['pos_v'] = pos_u[pos_edge_ids[:train_size]], pos_v[pos_edge_ids[:train_size]]
    dev_set['pos_u'], dev_set['pos_v'] = pos_u[pos_edge_ids[train_size:]], pos_v[pos_edge_ids[train_size:]]
    whole_set['pos_u'], whole_set['pos_v'] = pos_u[pos_edge_ids], pos_v[pos_edge_ids]

    # Sample negative cases
    # adj = sp.coo_matrix((np.ones(len(pos_u)), (pos_u, pos_v)))
    # adj_neg = 1 - adj.todense() - np.eye(number_of_edges)
    # neg_u, neg_v = np.where(adj_neg != 0)
    print(f"Sample negative cases")
    neg_u, neg_v = select_neg_edges(graph, graph.number_of_edges())
    train_set['neg_u'], train_set['neg_v'] = neg_u[neg_edge_ids[:train_size]], neg_v[neg_edge_ids[:train_size]]
    dev_set['neg_u'], dev_set['neg_v'] = neg_u[neg_edge_ids[train_size:]], neg_v[neg_edge_ids[train_size:]]
    whole_set['neg_u'], whole_set['neg_v'] = neg_u[neg_edge_ids], neg_v[neg_edge_ids]

    train_set['u'] = np.concatenate([train_set['pos_u'], train_set['neg_u']])
    train_set['v'] = np.concatenate([train_set['pos_v'], train_set['neg_v']])
    train_set['y'] = np.concatenate([np.ones(shape=len(train_set['pos_u']), dtype=int), np.zeros(shape=len(train_set['pos_u']), dtype=int)])

    dev_set['u'] = np.concatenate([dev_set['pos_u'], dev_set['neg_u']])
    dev_set['v'] = np.concatenate([dev_set['pos_v'], dev_set['neg_v']])
    dev_set['y'] = np.concatenate([np.ones(shape=len(dev_set['pos_u']), dtype=int), np.zeros(shape=len(dev_set['pos_u']), dtype=int)])

    test_set['u'] = evaluate_cases[:, 0]
    test_set['v'] = evaluate_cases[:, 1]
    test_set['y'] = - np.ones_like(test_set['u'])  # The label of test_set is unknown

    whole_set['u'] = np.concatenate([whole_set['pos_u'], whole_set['neg_u']])
    whole_set['v'] = np.concatenate([whole_set['pos_v'], whole_set['neg_v']])
    whole_set['y'] = np.concatenate([np.ones(shape=len(whole_set['pos_u']), dtype=int), np.zeros(shape=len(whole_set['pos_u']), dtype=int)])

    # Saving
    print(f"Saving to {OUTPUT_PATH}")
    np.savez(os.path.join(OUTPUT_PATH, f'{PREFIX}_train.npz'), **train_set)
    np.savez(os.path.join(OUTPUT_PATH, f'{PREFIX}_dev.npz'), **dev_set)
    np.savez(os.path.join(OUTPUT_PATH, f'{PREFIX}_test.npz'), **test_set)
    np.savez(os.path.join(OUTPUT_PATH, f'{PREFIX}_whole.npz'), **whole_set)

    with open(os.path.join(OUTPUT_PATH, f'{PREFIX}_train.pkl'), 'wb') as f:
        pickle.dump(train_set, f)
    with open(os.path.join(OUTPUT_PATH, f'{PREFIX}_dev.pkl'), 'wb') as f:
        pickle.dump(dev_set, f)
    with open(os.path.join(OUTPUT_PATH, f'{PREFIX}_test.pkl'), 'wb') as f:
        pickle.dump(test_set, f)
    with open(os.path.join(OUTPUT_PATH, f'{PREFIX}_whole.pkl'), 'wb') as f:
        pickle.dump(test_set, f)

    print(f"Generating Cache")
    [NetworkDataset(os.path.join(OUTPUT_PATH, f"{PREFIX}_{name}.pkl")) for name in ['train', 'dev', 'test', 'whole']]
    pass
