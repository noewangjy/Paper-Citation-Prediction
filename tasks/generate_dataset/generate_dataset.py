import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer


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
                if not j in G[i].keys() and i != j:
                    neg_u[idx], neg_v[idx] = i, j
                    pbar.update()
                    break

    return neg_u, neg_v


@torch.no_grad()
def generate_node_features(graph,
                           authors: np.ndarray,
                           authors_index: Dict[str, float],
                           abstracts: np.ndarray,
                           tokenizer: AutoTokenizer,
                           model: AutoModel,
                           device: torch.device = torch.device('cpu'),
                           num_author_features: int = 5,
                           num_abstract_features: int = 768,
                           batch_size: int = 32) -> np.ndarray:
    num_nodes: int = graph.number_of_nodes()
    authors_repr = torch.zeros(size=(num_nodes, num_author_features)).to(device)
    abstract_repr = torch.zeros(size=(num_nodes, num_abstract_features)).to(device)
    with tqdm.tqdm(range(num_nodes // batch_size + 1)) as pbar:
        for idx in range(num_nodes // batch_size + 1):
            idx_low = idx * batch_size
            idx_high = idx_low + batch_size

            abstract_tokens = torch.tensor(tokenizer(abstracts[idx_low:idx_high].tolist(), truncation=True, padding=True)['input_ids']).to(device)
            abstract_repr_vec = model(abstract_tokens, torch.zeros_like(abstract_tokens)).last_hidden_state[:, 0, :]
            abstract_repr[idx_low:idx_high] = abstract_repr_vec

            for i in range(min(num_author_features, len(authors[idx]))):
                authors_repr[idx][i] = authors_index[authors[idx][i]]
            pbar.update()

    return torch.cat([authors_repr, abstract_repr], dim=1).detach().cpu().numpy()


def generate_edge_features(dataset: Dict[str, np.ndarray],
                           graph,
                           num_edges: int) -> np.ndarray:
    edge_features = np.zeros(shape=(num_edges, 5), dtype=np.float32)
    with tqdm.tqdm(range(num_edges)) as pbar:
        for idx in range(num_edges):
            u, v, y = dataset['u'][idx], dataset['v'][idx], dataset['y'][idx]
            u_degree = int(graph.degree[u])
            v_degree = int(graph.degree[v])
            u_abstract = dataset['abstracts'][u]
            v_abstract = dataset['abstracts'][v]

            edge_features[idx] = v_degree + u_degree, \
                                 abs(v_degree - u_degree), \
                                 len(u_abstract) + len(v_abstract), \
                                 abs(len(u_abstract) + len(v_abstract)), \
                                 len(set(u_abstract.split()).intersection(set(v_abstract.split())))
            pbar.update()
    return edge_features


if __name__ == '__main__':
    # >>>>>> Beg modify parameters >>>>>>
    RECORD_NAMES: Dict[str, str] = {
        "abstracts": "abstracts.txt",
        "authors": "authors.txt",
        "authors_index": "authors.json",
        "edges": "edgelist.txt",
        "evaluate_cases": "test.txt"
    }
    DEV_RATIO: float = 0.2
    DATASET_PATH = input("Input the path of dataset") if len(sys.argv) < 2 else sys.argv[1]
    OUTPUT_PATH = input("Input the path of output") if len(sys.argv) < 3 else sys.argv[2]
    SEED = 0
    PREFIX: str = 'nullptr'
    GENERATE_NODE_FEATURE: bool = True
    GENERATE_EDGE_FEATURE: bool = False

    # <<<<<< End modify parameters <<<<<<
    if GENERATE_NODE_FEATURE:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        DEVICE: torch.device = torch.device('cuda:1')
        NUM_NODE_FEATURE_DIM: int = 8
        bert_model = AutoModel.from_pretrained("bert-base-uncased", num_labels=NUM_NODE_FEATURE_DIM)
        bert_model.to(DEVICE)
    np.random.seed(SEED)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # whole_set: Dict[str, np.ndarray] = {}
    # train_set: Dict[str, np.ndarray] = {}
    # dev_set: Dict[str, np.ndarray] = {}
    # test_set: Dict[str, np.ndarray] = {}
    datasets: Dict[str, Dict[str, np.ndarray]] = {
        'train': {},
        'dev': {},
        'whole': {},
        'test': {}
    }

    # Assign common information
    print(f"Reading dataset")
    dataset_path: str = DATASET_PATH
    graph: nx.Graph = read_graph(os.path.join(DATASET_PATH, RECORD_NAMES['edges']))
    abstracts: np.ndarray = read_abstracts(os.path.join(DATASET_PATH, RECORD_NAMES['abstracts']))
    authors: np.ndarray = read_authors(os.path.join(DATASET_PATH, RECORD_NAMES['authors']))
    authors_index: Dict[str, float] = json.load(open(os.path.join(DATASET_PATH, RECORD_NAMES['authors_index'])))
    assert len(authors) == len(abstracts) == len(graph), ValueError(f"Data not aligned")
    evaluate_cases: np.ndarray = read_evaluate_cases(os.path.join(DATASET_PATH, RECORD_NAMES['evaluate_cases']))
    edges: np.ndarray = np.array(graph.edges())

    for name in datasets.keys():
        datasets[name]['abstracts'], datasets[name]['authors'], datasets[name]['origin_edges'] = abstracts, authors, edges

    # Build edge features
    if GENERATE_NODE_FEATURE:
        print(f"Generating node features")
        node_features = generate_node_features(graph,
                                               authors,
                                               authors_index,
                                               abstracts,
                                               tokenizer,
                                               bert_model,
                                               device=DEVICE)
        for name in datasets.keys():
            datasets[name]['node_features'] = node_features

    # Generate ids
    print(f"Generating IDs")
    number_of_edges: int = graph.number_of_edges()
    pos_edge_ids = np.arange(number_of_edges)
    pos_edge_ids = np.random.permutation(pos_edge_ids)
    neg_edge_ids = np.arange(number_of_edges)
    neg_edge_ids = np.random.permutation(neg_edge_ids)

    # Calculate test size and train size
    dev_size = int(number_of_edges * DEV_RATIO)
    train_size = number_of_edges - dev_size

    # Sample positive cases
    print(f"Sampling positive cases")
    pos_u, pos_v = edges[:, 0], edges[:, 1]
    datasets['train']['pos_u'], datasets['train']['pos_v'] = pos_u[pos_edge_ids[:train_size]], pos_v[pos_edge_ids[:train_size]]
    datasets['dev']['pos_u'], datasets['dev']['pos_v'] = pos_u[pos_edge_ids[train_size:]], pos_v[pos_edge_ids[train_size:]]
    datasets['whole']['pos_u'], datasets['whole']['pos_v'] = pos_u[pos_edge_ids], pos_v[pos_edge_ids]

    # Sample negative cases
    # adj = sp.coo_matrix((np.ones(len(pos_u)), (pos_u, pos_v)))
    # adj_neg = 1 - adj.todense() - np.eye(number_of_edges)
    # neg_u, neg_v = np.where(adj_neg != 0)
    print(f"Sampling negative cases")
    neg_u, neg_v = select_neg_edges(graph, graph.number_of_edges())
    datasets['train']['neg_u'], datasets['train']['neg_v'] = neg_u[neg_edge_ids[:train_size]], neg_v[neg_edge_ids[:train_size]]
    datasets['dev']['neg_u'], datasets['dev']['neg_v'] = neg_u[neg_edge_ids[train_size:]], neg_v[neg_edge_ids[train_size:]]
    datasets['whole']['neg_u'], datasets['whole']['neg_v'] = neg_u[neg_edge_ids], neg_v[neg_edge_ids]

    # Concat positive and negative cases
    for name in ['train', 'dev', 'whole']:
        datasets[name]['u'] = np.concatenate([datasets[name]['pos_u'], datasets[name]['neg_u']])
        datasets[name]['v'] = np.concatenate([datasets[name]['pos_v'], datasets[name]['neg_v']])
        datasets[name]['y'] = np.concatenate([np.ones(shape=len(datasets[name]['pos_u']), dtype=int), np.zeros(shape=len(datasets[name]['pos_u']), dtype=int)])

    datasets['test']['u'] = evaluate_cases[:, 0]
    datasets['test']['v'] = evaluate_cases[:, 1]
    datasets['test']['y'] = - np.ones_like(datasets['test']['u'])  # The label of test_set is unknown

    # Build edge features
    if GENERATE_EDGE_FEATURE:
        print(f"Generating edge features")
        for name in datasets.keys():
            datasets[name]['edge_features'] = generate_edge_features(datasets[name], graph, len(datasets[name]['u']))

    # Saving
    print(f"Saving to {OUTPUT_PATH}")
    for name in datasets.keys():
        with open(os.path.join(OUTPUT_PATH, f'{PREFIX}_{name}.pkl'), 'wb') as f:
            pickle.dump(datasets[name], f)

    print(f"Generating Cache")
    pass
