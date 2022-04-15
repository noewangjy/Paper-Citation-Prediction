import math
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
import tqdm

from src.utils.fitter import fit_lr_classifier, infer_lr_classifier
from src.utils.io import check_md5
from src.utils.io import split_graph
from src.utils.submission import generate_submission


def compute_distance(graph: dgl.DGLGraph, u, v, max_distance=0x10):
    u_bfs_nodes = dgl.bfs_nodes_generator(graph, u)
    for idx, nodes in zip(range(0, max_distance), u_bfs_nodes):
        if v in nodes:
            return idx
    return max_distance


def compute_pagerank(graph: dgl.DGLGraph,
                     num_loops: int = 20,
                     DAMP: float = 0.85):
    N = graph.number_of_nodes()
    graph.ndata['pv'] = torch.ones(N, device=graph.device) / N
    degrees = graph.out_degrees(graph.nodes()).type(torch.float32)
    with tqdm.tqdm(range(num_loops)) as pbar:
        pbar.set_description(f"Running PageRank")
        for k in range(num_loops):
            graph.ndata['pv'] = graph.ndata['pv'] / degrees
            graph.update_all(message_func=fn.copy_src(src='pv', out='m'),
                             reduce_func=fn.sum(msg='m', out='pv'))
            graph.ndata['pv'] = (1 - DAMP) / N + DAMP * graph.ndata['pv']
            pbar.update()
    return graph.ndata['pv']


def shortest_path_safe(graph, u, v, max_distance=0xF):
    if v not in graph[u].keys():
        try:
            length = nx.shortest_path_length(graph, u, v)
            return length
        except nx.NetworkXNoPath as e:
            return max_distance
    else:
        for i in graph[u].keys():
            try:
                return nx.shortest_path_length(graph, i, v) + 1
            except nx.NetworkXNoPath as e:
                pass
        return max_distance


def compute_shortest_path(graph: nx.Graph, u, v, max_distance=0xF):
    if v in graph[u].keys():
        neighbors = set(graph[u].keys())
        neighbors.remove(v)
        return min([shortest_path_safe(graph, i, v) for i in neighbors]) + 1
    else:
        return shortest_path_safe(graph, u, v, max_distance)


def get_feature(u, v,
                authors: List[List[str]],
                authors_index_mapping: Dict[str, int],
                authors_graph: nx.Graph,
                authors_graph_weights,
                page_rank):
    u_authors_enc, v_authors_enc = [authors_index_mapping[i] for i in authors[u]], [authors_index_mapping[i] for i in authors[v]]

    u_author, v_author = u_authors_enc[0], v_authors_enc[0]
    u_author_neighbors, v_author_neighbors = set(authors_graph[u_author].keys()), set(authors_graph[v_author].keys())
    common_neighbors = u_author_neighbors.intersection(v_author_neighbors)
    union_neighbors = u_author_neighbors.union(v_author_neighbors)
    u_weight_indices = [authors_graph.get_edge_data(u_author, i)[0]['id'] for i in authors_graph[u_author].keys()]
    v_weight_indices = [authors_graph.get_edge_data(v_author, i)[0]['id'] for i in authors_graph[v_author].keys()]

    shortest_path = shortest_path_safe(authors_graph, u_author, v_author)
    common_neighbors_coef: int = len(common_neighbors)
    # num_u_neighbor = len(u_author_neighbors)
    # num_v_neighbor = len(v_author_neighbors)
    jaccard_coef: float = common_neighbors_coef / (1e-8 + len(union_neighbors))
    # adar: float = sum(map(lambda x: 1 / math.log1p(1e-8 + len(authors_graph[x].keys())), common_neighbors))
    # pref_attachment: int = len(u_author_neighbors) * len(v_author_neighbors)
    u_link_value = sum([authors_graph_weights[i] for i in u_weight_indices])
    v_link_value = sum([authors_graph_weights[i] for i in v_weight_indices])
    rank_min = float(min(page_rank[u_author], page_rank[v_author]))
    rank_max = float(max(page_rank[u_author], page_rank[v_author]))
    # return shortest_path, \ # 0
    #        common_neighbors_coef, \ # 1
    #        num_u_neighbor, \ # 2
    #        num_v_neighbor, \ # 3
    #        jaccard_coef, \ # 4
    #        adar, \ # 5
    #        pref_attachment, \ # 6
    #        rank_min, \ # 7
    #        rank_max, \ # 8
    #        min(u_link_value, v_link_value), \ # 9
    #        max(u_link_value, v_link_value) # 10
    return shortest_path, jaccard_coef, rank_min, rank_max, min(u_link_value, v_link_value), max(u_link_value, v_link_value)


def get_feature_array(u_array: np.ndarray,
                      v_array: np.ndarray,
                      num_features: int,
                      authors: List[List[str]],
                      authors_index_mapping: Dict[str, int],
                      authors_graph_nx: nx.Graph,
                      authors_graph_weights: np.ndarray,
                      pagerank: np.ndarray):
    res = np.zeros(shape=(len(u_array), num_features))

    with tqdm.tqdm(range(len(res))) as pbar:
        for idx in range(res.shape[0]):
            res[idx] = get_feature(u_array[idx], v_array[idx],
                                   authors,
                                   authors_index_mapping,
                                   authors_graph_nx,
                                   authors_graph_weights,
                                   pagerank)
            # if idx % 1e3 == 0 and idx != 0: pbar.update(1e3)
            pbar.update()
    return res


def f(u_array: np.ndarray,
      v_array: np.ndarray,
      num_features: int,
      authors: List[List[str]],
      authors_index_mapping: Dict[str, int],
      authors_graph_nx: nx.Graph,
      authors_graph_weights: np.ndarray,
      pagerank: np.ndarray):
    print("hi")
    return np.array([1, 2])


def get_feature_array_mp(u_array,
                         v_array,
                         num_features,
                         authors,
                         authors_index_mapping,
                         authors_graph_nx,
                         authors_graph_weights,
                         pagerank,
                         num_procs: int = 12):
    num_edges = len(u_array)
    batch_size: int = math.ceil(num_edges / num_procs)

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork")) as pool:
        results = []
        for idx in range(num_procs):
            results.append(pool.submit(get_feature_array,
                                       u_array[idx * batch_size: (idx + 1) * batch_size],
                                       v_array[idx * batch_size: (idx + 1) * batch_size],
                                       num_features,
                                       authors,
                                       authors_index_mapping,
                                       authors_graph_nx,
                                       authors_graph_weights,
                                       pagerank))
            # results.append(pool.submit(f,
            #                            u_array[idx * batch_size: (idx + 1) * batch_size],
            #                            v_array[idx * batch_size: (idx + 1) * batch_size],
            #                            num_features,
            #                            authors,
            #                            authors_index_mapping,
            #                            authors_graph_nx,
            #                            authors_graph_weights,
            #                            pagerank))
    features = np.concatenate(list(map(lambda x: x.result(), results)))
    return features


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    ## Load dataset
    whole_dataset_path = '../../data/converted/nullptr_graph_conv_whole.pkl'
    train_dataset_path = '../../data/converted/nullptr_graph_conv_train.pkl'
    dev_dataset_path = '../../data/converted/nullptr_graph_conv_dev.pkl'
    test_dataset_path = '../../data/converted/nullptr_graph_conv_test.pkl'
    uv_list_path = '../../data/neo_converted/uv_list.pkl'
    print(check_md5(uv_list_path))

    whole_dataset = pickle.load(open(whole_dataset_path, 'rb'))
    train_dataset = pickle.load(open(train_dataset_path, 'rb'))
    dev_dataset = pickle.load(open(dev_dataset_path, 'rb'))
    test_dataset = pickle.load(open(test_dataset_path, 'rb'))
    uv_dataset = pickle.load(open(uv_list_path, 'rb'))

    authors_graph_edges = whole_dataset['authors_graph_edges']
    authors_graph_weights = whole_dataset['authors_graph_weights']
    authors_mapping = whole_dataset['authors_index_mapping']

    # GraphSAGE on authors network
    ctx = split_graph(authors_graph_edges, device=device)
    page_rank = compute_pagerank(ctx['whole_graph'], num_loops=100).detach().cpu().numpy()

    whole_authors_graph = ctx['whole_graph']
    whole_authors_graph.edata['weight'] = torch.tensor(whole_dataset['authors_graph_weights']).to(whole_authors_graph.device)
    authors_graph_nx = whole_authors_graph.cpu().to_networkx()

    X_train = get_feature_array(uv_dataset['train_u'],
                                uv_dataset['train_v'],
                                6,
                                whole_dataset['authors'],
                                whole_dataset['authors_index_mapping'],
                                authors_graph_nx,
                                authors_graph_weights,
                                page_rank)  # Loss: 0.2263460069674607, Accuracy: 0.9180460733271976, F1-score: 0.9205102126922513, using 0,4,7,8,9,10
    X_dev = get_feature_array(uv_dataset['dev_u'],
                              uv_dataset['dev_v'],
                              6,
                              whole_dataset['authors'],
                              whole_dataset['authors_index_mapping'],
                              authors_graph_nx,
                              authors_graph_weights,
                              page_rank)
    X_test = get_feature_array(uv_dataset['test_u'],
                               uv_dataset['test_v'],
                               6,
                               whole_dataset['authors'],
                               whole_dataset['authors_index_mapping'],
                               authors_graph_nx,
                               authors_graph_weights,
                               page_rank)

    Y_train = uv_dataset['train_y']
    Y_dev = uv_dataset['dev_y']

    clf = fit_lr_classifier(X_train, Y_train, X_dev, Y_dev, max_iter=200)

    scores = infer_lr_classifier(clf, X_test)
    generate_submission('./outputs', scores, "author_graph_lr")
    features = {
        '__text__': 'shortest_path, jaccard_coef, rank_min, rank_max, weight_min, weight_max',
        'uv_chksum': check_md5(uv_list_path),
        'X_train': X_train,
        'Y_train': Y_train,
        'X_dev': X_dev,
        'Y_dev': Y_dev,
        'X_test': X_test,
    }
    with open('author_graph_lr_features.pkl', 'wb') as f:
        pickle.dump(features, f)
