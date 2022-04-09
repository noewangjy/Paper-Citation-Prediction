import logging

import networkx as nx
from typing import Tuple, Dict, List, Any, Union, Callable
import numpy as np
import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import json
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import nltk


def feature_fn_graph(cfg, context, datasets, logger: logging.Logger):
    for dataset_name, dataset in datasets.items():
        dataset['graph'] = context['graph']
    return datasets


def feature_fn_edges(cfg, context, datasets, logger: logging.Logger):
    for dataset_name, dataset in datasets.items():
        dataset['origin_edges'] = context['graph'].edges()
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
                u_abstract = dataset['abstracts'][u]
                v_abstract = dataset['abstracts'][v]

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
                    features[idx][vocab[word]] = 1
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
    acc = set()
    for authors in context['authors']:
        acc.update(set(authors))

    authors_list: List[str] = list(acc)
    authors_list.sort()
    values = np.arange(len(authors_list)) / len(authors_list) + 1 / len(authors_list)
    for dataset_name, dataset in datasets.items():
        dataset['authors_index'] = values
    return datasets
