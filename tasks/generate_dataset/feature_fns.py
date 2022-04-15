import itertools
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, List

import dgl
import networkx as nx
import nltk
import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer


def feature_fn_graph(cfg, context, datasets, logger: logging.Logger):
    for dataset_name, dataset in datasets.items():
        dataset['graph'] = context['graph']
    return datasets


def feature_fn_edges(cfg, context, datasets, logger: logging.Logger):
    for dataset_name, dataset in datasets.items():
        dataset['origin_edges'] = np.array(context['graph'].edges())
    return datasets


def feature_fn_authors(cfg, context, datasets, logger: logging.Logger):
    for dataset_name, dataset in datasets.items():
        dataset['authors'] = context['authors']
    return datasets


def feature_fn_abstracts(cfg, context, datasets, logger: logging.Logger):
    for dataset_name, dataset in datasets.items():
        dataset['abstracts'] = context['abstracts']
    return datasets


def feature_fn_pos_uv(cfg, context, datasets, logger: logging.Logger):
    logger.info('feature_fn_pos_uv')

    edges = np.array(context['graph'].edges())
    pos_u, pos_v = edges[:, 0], edges[:, 1]
    for dataset_name, dataset in datasets.items():

        if dataset_name == 'train':
            dataset['pos_u'], dataset['pos_v'] = pos_u[context['ids']['pos'][:context['size']['train']]], pos_v[context['ids']['pos'][:context['size']['train']]]
        elif dataset_name == 'dev':
            dataset['pos_u'], dataset['pos_v'] = pos_u[context['ids']['pos'][context['size']['train']:]], pos_v[context['ids']['pos'][context['size']['train']:]]
        elif dataset_name == 'whole':
            dataset['pos_u'], dataset['pos_v'] = pos_u[context['ids']['pos']], pos_v[context['ids']['pos']]

        if 'u' in dataset.keys():
            dataset['u'] = np.concatenate([dataset['u'], dataset['pos_u']])
        else:
            dataset['u'] = dataset['pos_u']

        if 'v' in dataset.keys():
            dataset['v'] = np.concatenate([dataset['v'], dataset['pos_v']])
        else:
            dataset['v'] = dataset['pos_v']

        if 'y' in dataset.keys():
            dataset['y'] = np.concatenate([dataset['y'], np.ones_like(dataset['pos_u'])])
        else:
            dataset['y'] = np.ones_like(dataset['pos_u'])
    return datasets


def _select_neg_edges(idx,
                      G: nx.Graph,
                      num_neg_edges: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(idx):
        np.random.rand(1)  # Change Seed
    num_nodes = G.number_of_nodes()
    neg_u, neg_v = np.zeros(shape=(num_neg_edges,), dtype=int), np.zeros(shape=(num_neg_edges,), dtype=int)
    with tqdm.tqdm(range(num_neg_edges)) as pbar:
        pbar.set_description(f"Idx: {idx}")
        for idx in range(num_neg_edges):
            while True:
                i, j = np.random.randint(0, num_nodes), np.random.randint(0, num_nodes)
                if not j in G[i].keys() and i != j:
                    neg_u[idx], neg_v[idx] = i, j
                    break
            if idx % 1e4 == 0: pbar.update(1e4)
    return neg_u, neg_v


def feature_fn_neg_uv(cfg, context, datasets, logger: logging.Logger):
    logger.info('feature_fn_neg_uv')

    # num_neg_edges = context['size']['whole']
    # num_procs: int = max(cfg.target_features.neg_uv.num_procs, 1) if cfg.target_features.neg_uv is not None else 4
    # batch_size: int = math.ceil(num_neg_edges / num_procs)
    # worker_load = [batch_size] * (num_procs - 1) + [num_neg_edges - (num_procs - 1) * batch_size]
    # with ProcessPoolExecutor(max_workers=num_procs) as pool:
    #     results = []
    #     for idx in range(num_procs):
    #         results.append(pool.submit(select_neg_edges, idx, context['graph'], worker_load[idx]))
    #         np.random.rand(1)
    #     pool.shutdown()
    # neg_u = np.concatenate([item.result()[0] for item in results])
    # neg_v = np.concatenate([item.result()[1] for item in results])
    # @warning Should not use multiprocessing until pseudo random problem is solved
    np.random.seed(int(cfg.random_seed))

    neg_u, neg_v = _select_neg_edges(0, context['graph'], context['size']['whole'])

    for dataset_name, dataset in datasets.items():
        if dataset_name == 'train':
            dataset['neg_u'], dataset['neg_v'] = neg_u[context['ids']['neg'][:context['size']['train']]], neg_v[context['ids']['neg'][:context['size']['train']]]
        elif dataset_name == 'dev':
            dataset['neg_u'], dataset['neg_v'] = neg_u[context['ids']['neg'][context['size']['train']:]], neg_v[context['ids']['neg'][context['size']['train']:]]
        elif dataset_name == 'whole':
            dataset['neg_u'], dataset['neg_v'] = neg_u[context['ids']['neg']], neg_v[context['ids']['neg']]

        if 'u' in dataset.keys():
            dataset['u'] = np.concatenate([dataset['u'], dataset['neg_u']])
        else:
            dataset['u'] = dataset['neg_u']

        if 'v' in dataset.keys():
            dataset['v'] = np.concatenate([dataset['v'], dataset['neg_v']])
        else:
            dataset['v'] = dataset['neg_v']

        if 'y' in dataset.keys():
            dataset['y'] = np.concatenate([dataset['y'], np.zeros_like(dataset['neg_u'])])
        else:
            dataset['y'] = np.zeros_like(dataset['neg_u'])
    return datasets


@torch.no_grad()
def feature_fn_node_abstract_bert(cfg, context, datasets, logger: logging.Logger):
    bert_args = cfg.target_features.node_abstract_bert
    num_abstract_features = bert_args.node_abstract_bert
    batch_size = bert_args.batch_size
    num_node_feature_dim = bert_args.num_node_feature_dim

    tokenizer = AutoTokenizer.from_pretrained("bert.yaml-base-uncased")
    bert_model = AutoModel.from_pretrained("bert.yaml-base-uncased", num_labels=num_node_feature_dim)
    device: torch.device = torch.device('cuda:1')
    bert_model.to(device)

    graph = context['graph']
    abstracts = context['abstracts']
    num_nodes: int = graph.number_of_nodes()

    abstract_repr = torch.zeros(size=(num_nodes, num_abstract_features)).to(device)
    with tqdm.tqdm(range(num_nodes // batch_size + 1)) as pbar:
        for idx in range(num_nodes // batch_size + 1):
            idx_low = idx * batch_size
            idx_high = idx_low + batch_size

            abstract_tokens = torch.tensor(tokenizer(abstracts[idx_low:idx_high].tolist(), truncation=True, padding=True)['input_ids']).to(device)
            abstract_repr_vec = bert_model(abstract_tokens, torch.zeros_like(abstract_tokens)).last_hidden_state[:, 0, :]
            abstract_repr[idx_low:idx_high] = abstract_repr_vec

            pbar.update()

    abstract_repr_np = abstract_repr.detach().cpu().numpy()
    for dataset_name, dataset in datasets.items():
        dataset['node_abstract_bert'] = abstract_repr_np

    return datasets


def feature_fn_edge_generic(cfg, context, datasets, logger: logging.Logger):
    graph = context['graph']

    for dataset_name, dataset in datasets.items():
        num_edges = len(dataset['u'])
        edge_features = np.zeros(shape=(num_edges, 5), dtype=np.float32)
        with tqdm.tqdm(range(num_edges)) as pbar:
            for idx in range(num_edges):
                u, v, y = dataset['u'][idx], dataset['v'][idx], dataset['y'][idx]
                u_degree = int(graph.degree[u])
                v_degree = int(graph.degree[v])
                u_abstract = context['abstracts'][u]
                v_abstract = context['abstracts'][v]

                edge_features[idx] = v_degree + u_degree, \
                                     abs(v_degree - u_degree), \
                                     len(u_abstract) + len(v_abstract), \
                                     abs(len(u_abstract) + len(v_abstract)), \
                                     len(set(u_abstract.split()).intersection(set(v_abstract.split())))
                pbar.update()
        dataset['edge_generic'] = edge_features
    return datasets


class _CustomTokenizer:
    def __init__(self):
        self.regex_tokenizer = nltk.RegexpTokenizer(r'\w+')

    def __call__(self, string):
        return list(map(lambda x: nltk.PorterStemmer().stem(x), self.regex_tokenizer.tokenize(string)))


def _tokenize_fn(idx,
                 abstract_list: List[str],
                 interpunctuations: set,
                 stops: set,
                 prohibited_characters: set,
                 prohibited_words: set) -> Dict[str, int]:
    def judge(word: str,
              interpunctuations: set,
              stops,
              prohibited_characters,
              prohibited_words):
        return (word not in interpunctuations) and (word not in stops) and (not any([i in prohibited_characters for i in set(word)])) and (word not in prohibited_words)

    # ProcessPoolExecuter require all args to be serializable, while nltk.PorterStemmer and nltk.RegexTokenizer are not
    # The tokenizer must instance in function
    tokenizer = _CustomTokenizer()
    num_abstracts: int = len(abstract_list)
    res: Dict[str, int] = {}
    with tqdm.tqdm(range(num_abstracts)) as pbar:
        pbar.set_description(f"Idx: {idx}")
        for idx in range(num_abstracts):
            for word in tokenizer(abstract_list[idx]):
                if judge(word, interpunctuations, stops, prohibited_characters, prohibited_words):
                    if word in res.keys():
                        res[word] += 1
                    else:
                        res[word] = 1
            if idx % 1e2 == 0:
                pbar.update(1e2)

    return res


def _generate_frequency_dict(cora_like_cfg, abstracts: List[str], num_procs: int = 1) -> Dict[str, int]:
    from nltk.corpus import stopwords
    INTERPUNCTUATIONS = {',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%'}
    STOPS = set(stopwords.words("english"))
    PROHIBITED_WORDS: set = set(['-', '--', '‐'] + [str(i) for i in range(10000)])
    PROHIBITED_CHARACTERS: set = INTERPUNCTUATIONS | \
                                 set([chr(idx) for idx in range(0x7f, 0x4000)]) | \
                                 {'`', "'", '"', '\\', '~', '_', '|', '/', '<', '>', '{', '}', '=', '+', '^', '’', '“', '”', '‘', }
    MAX_FREQ = cora_like_cfg.max_freq
    MIN_FREQ = cora_like_cfg.min_freq
    num_abstracts: int = len(abstracts)
    batch_size: int = math.ceil(num_abstracts / num_procs)

    # Debug lines that can be deleted in next commit
    # results = []
    # for idx in range(num_procs):
    #     results.append(tokenize_fn(idx, abstracts[idx * batch_size: (idx + 1) * batch_size], tokenizer, INTERPUNCTUATIONS, STOPS, PROHIBITED_CHARACTERS, PROHIBITED_WORDS))
    with ProcessPoolExecutor(max_workers=num_procs) as pool:
        results = []
        for idx in range(num_procs):
            results.append(pool.submit(_tokenize_fn, idx, abstracts[idx * batch_size: (idx + 1) * batch_size], INTERPUNCTUATIONS, STOPS, PROHIBITED_CHARACTERS, PROHIBITED_WORDS))
        # pool.shutdown()

    frequency_dict_reduced = {}
    with tqdm.tqdm(range(len(results))) as pbar:
        pbar.set_description(f"Reducing")
        for frequency_dict in map(lambda x: x.result(), results):
            for key in frequency_dict:
                if key in frequency_dict_reduced:
                    frequency_dict_reduced[key] += frequency_dict[key]
                else:
                    frequency_dict_reduced[key] = frequency_dict[key]
            pbar.update()

    frequency_dict_reduced_filtered = {k: v for k, v in frequency_dict_reduced.items() if MAX_FREQ > v > MIN_FREQ}

    return frequency_dict_reduced_filtered


def feature_fn_cora_like(cfg, context, datasets, logger: logging.Logger):
    logger.info('feature_fn_cora_like')

    abstracts: List[str] = context['abstracts']
    frequency_dict: Dict[str, int] = _generate_frequency_dict(cfg.target_features.cora_like, abstracts, num_procs=max(cfg.target_features.cora_like.num_procs, 1))
    features = np.zeros(shape=(len(abstracts), len(frequency_dict) + 1), dtype=np.uint8)
    print("Estimated feature shape: ", features.shape)
    # features[:, -1] = 1  # Last column is always 1
    word_index: List[str] = sorted(list(frequency_dict.keys()))
    vocab = {k: v for v, k in enumerate(word_index)}

    regex_tokenizer = _CustomTokenizer()
    num_abstracts: int = len(abstracts)
    with tqdm.tqdm(range(num_abstracts)) as pbar:
        pbar.set_description("Creating abstract feature")
        for idx in range(num_abstracts):
            for word in regex_tokenizer(abstracts[idx]):
                if word in vocab.keys():
                    features[idx][vocab[word]] += 1
            if features[idx].sum() <= 0:
                features[idx][-1] = 1  # Last column is 1 -> No keyword
            if idx % 1e2 == 0:
                pbar.update(1e2)

    for dataset_name, dataset in datasets.items():
        dataset['cora_features'] = features
        dataset['cora_vocab'] = vocab
        dataset['cora_word_index'] = word_index
    return datasets


def feature_fn_authors_index(cfg, context, datasets, logger):
    logger.info('feature_fn_authors_index')
    acc = {}
    for authors in context['authors']:
        for author in authors:
            if author in acc.keys():
                acc[author] += 1
            else:
                acc[author] = 1

    authors_index_names: List[str] = list(acc.keys())
    authors_index_names.sort()

    authors_index_freq = [acc[author] for author in authors_index_names]
    num_authors = len(authors_index_names)
    authors_index_mapping = {k: v for k, v in zip(authors_index_names, np.arange(0, num_authors))}
    for dataset_name, dataset in datasets.items():
        dataset['authors_index_names'] = np.array(authors_index_names, dtype=object)
        dataset['authors_index_mapping'] = authors_index_mapping
        dataset['authors_index_freq'] = np.array(authors_index_freq, dtype=int)
    return datasets


def feature_fn_authors_graph(cfg, context, datasets, logger):
    logger.info('feature_fn_authors_graph')
    if 'authors_index_names' not in datasets['whole'].keys():
        datasets = feature_fn_authors_index(cfg, context, datasets, logger)
    if 'origin_edges' not in datasets['whole'].keys():
        datasets = feature_fn_edges(cfg, context, datasets, logger)

    num_authors: int = len(datasets['whole']['authors_index_names'])
    authors_list: List[str] = datasets['whole']['authors_index_names']

    author_mapping: Dict[str, int] = {author: idx for author, idx in zip(authors_list, np.arange(0, num_authors))}
    authors: List[List[str]] = context['authors']
    weighted_edges: Dict[Tuple[int, int], int] = {}
    with tqdm.tqdm(range(num_authors)) as pbar:
        for idx, author_group in enumerate(authors):
            for author_idx_tuple in map(lambda x: (author_mapping[x[0]], author_mapping[x[1]]), itertools.combinations(author_group, 2)):
                if author_idx_tuple in weighted_edges.keys():
                    weighted_edges[author_idx_tuple] += cfg.target_features.authors_graph.alpha
                else:
                    weighted_edges[author_idx_tuple] = cfg.target_features.authors_graph.alpha
            if idx % 1e4 == 0: pbar.update(1e4)

    num_edges = len(datasets['whole']['origin_edges'])
    edges = datasets['whole']['origin_edges']
    with tqdm.tqdm(range(num_edges)) as pbar:
        for edge_idx in range(num_edges):
            u, v = edges[edge_idx]
            for author_idx_tuple in map(lambda x: (author_mapping[x[0]], author_mapping[x[1]]), [(i, j) for i in authors[u] for j in authors[v]]):
                if author_idx_tuple in weighted_edges.keys():
                    weighted_edges[author_idx_tuple] += cfg.target_features.authors_graph.beta
                else:
                    weighted_edges[author_idx_tuple] = cfg.target_features.authors_graph.beta
            if edge_idx % 1e4 == 0: pbar.update(1e4)

    unweighted_edges, edge_weights = np.array(list(map(lambda x: list(x), weighted_edges.keys()))), np.array(list(weighted_edges.values()))

    # Graph construction example
    # G = nx.Graph()
    # G.add_weighted_edges_from(np.concatenate([unweighted_edges, np.expand_dims(edge_weights, axis=1)], axis=1))
    # G = dgl.from_networkx(nx.Graph())
    # G.add_edges(torch.tensor(unweighted_edges[:,0]),torch.tensor(unweighted_edges[:,1]), {'w': torch.tensor(edge_weights)})
    for dataset_name, dataset in datasets.items():
        dataset['authors_graph_edges'] = unweighted_edges
        dataset['authors_graph_weights'] = edge_weights
    return datasets


def _compute_distance(graph: dgl.DGLGraph, u_authors_enc: List[int], v_authors_enc: List[int], max_distance=0x10):
    """
    Distance = np(-x) where x = number of intersections between u_author_neighbors and v_author_neighbors
    :param graph:
    :param u_authors_enc:
    :param v_authors_enc:
    :param max_distance:
    :return:
    """
    u_bfs_nodes, v_bfs_nodes = dgl.bfs_nodes_generator(graph, u_authors_enc), dgl.bfs_nodes_generator(graph, v_authors_enc)
    distance = min(len(u_bfs_nodes), len(v_bfs_nodes), max_distance)
    num_intersections = len(np.intersect1d(torch.cat(u_bfs_nodes[0: distance]).detach().cpu().numpy(),
                                           torch.cat(v_bfs_nodes[0: distance]).detach().cpu().numpy()))
    return np.exp(-num_intersections)


def _compute_batch_distance(proc_id: int,
                            graph: dgl.DGLGraph,
                            authors: np.ndarray,
                            authors_mapping: Dict[str, int],
                            u_array: np.ndarray,
                            v_array: np.ndarray,
                            max_distance) -> np.ndarray:
    num_edges = len(u_array)
    distance_features = np.zeros_like(u_array, dtype=np.float)
    with tqdm.tqdm(range(num_edges)) as pbar:
        pbar.set_description(f"Idx: {proc_id}")
        for idx, u, v in zip(range(num_edges), u_array, v_array):
            u_authors_enc, v_authors_enc = [authors_mapping[item] for item in authors[u]], [authors_mapping[item] for item in authors[v]]
            distance_features[idx] = _compute_distance(graph, u_authors_enc, v_authors_enc, max_distance)
            if idx % 1e2 == 0: pbar.update(1e2)

    return distance_features


#
# def _compute_distance_memory(graph: dgl.DGLGraph, u_array: torch.Tensor, v_array: torch.Tensor, max_distance: int):
#     num_edges = len(u_array)
#     distance_memory: torch.Tensor = torch.zeros_like(u_array)
#     with tqdm.tqdm(range(num_edges)) as pbar:
#         for idx, u, v in zip(range(num_edges), u_array, v_array):
#             distance_memory[idx] = _compute_distance(graph, u, v, max_distance)
#             if idx % 1000 == 0: pbar.update(1000)
#
#     return distance_memory.detach().cpu().numpy()


def feature_fn_essay_distance(cfg, context, datasets, logger):
    logger.info('feature_fn_essay_distance')
    if 'u' not in datasets['whole'].keys():
        logger.warning("Dataset has not initialized u/v, enable uv by default")
        datasets = feature_fn_pos_uv(cfg, context, datasets, logger)
        datasets = feature_fn_neg_uv(cfg, context, datasets, logger)
    if 'authors_graph_edges' not in datasets['whole'].keys():
        logger.warning("Dataset has not initialized authors_graph, enable authors_grap by default")
        datasets = feature_fn_authors_graph(cfg, context, datasets, logger)

    logger.info(f'generating dgl graph')
    unweighted_edges: np.ndarray = datasets['whole']['authors_graph_edges']
    authors: np.ndarray = context['authors']
    authors_mapping: Dict[str, int] = datasets['whole']['authors_index_mapping']
    graph: dgl.DGLGraph = dgl.graph((torch.tensor(unweighted_edges[:, 0], dtype=torch.int32), torch.tensor(unweighted_edges[:, 1], dtype=torch.int32)), idtype=torch.int32)

    # logger.info('computing node distances')
    # distance_memory: np.ndarray = _compute_distance_memory(graph,
    #                                                        torch.tensor(datasets['whole']['pos_u'], dtype=torch.int32),
    #                                                        torch.tensor(datasets['whole']['pos_v'], dtype=torch.int32),
    #                                                        cfg.target_features.essay_distance.max_distance)
    # res = _compute_batch_distance(0, graph, authors, authors_mapping, datasets['whole']['u'], datasets['whole']['v'], cfg.target_features.essay_distance.max_distance)
    num_edges = len(datasets['whole']['u'])
    num_procs: int = cfg.target_features.essay_distance.num_procs
    batch_size: int = math.ceil(num_edges / num_procs)
    with ProcessPoolExecutor(max_workers=num_procs) as pool:
        results = []
        for idx in range(num_procs):
            results.append(pool.submit(_compute_batch_distance,
                                       idx,
                                       graph,
                                       authors,
                                       authors_mapping,
                                       datasets['whole']['u'][idx * batch_size: (idx + 1) * batch_size],
                                       datasets['whole']['v'][idx * batch_size: (idx + 1) * batch_size],
                                       cfg.target_features.essay_distance.max_distance))
        # pool.shutdown()
    distance_features = np.concatenate(list(map(lambda x: x.result(), results)))

    logger.info(f'finalizing distance result')
    # distance_memory = np.array([distance_map[(u, v)] if (u, v) in distance_map.keys() else cfg.target_features.essay_distance.max_distance - 1 for u, v in zip(datasets['whole']['pos_u'], datasets['whole']['pos_v'])])
    for dataset_name, dataset in datasets.items():
        dataset['essay_distance'] = distance_features
    return datasets
