import logging

import hydra
from hydra.utils import to_absolute_path
import numpy as np
import networkx as nx
import scipy.sparse as sp
from random import randint
import dgl
import time
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gensim.models import Word2Vec

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

logger = logging.getLogger(__name__)


def random_walk(G, node, walk_length):
    # Simulates a random walk of length "walk_length" starting from node "node"

    # Please insert your code for Task 1 here
    walk = [node]
    for i in range(walk_length):
        neighbors = list(G.neighbors(walk[i]))
        walk.append(neighbors[randint(0, len(neighbors) - 1)])

    walk = [str(node) for node in walk]
    return walk


# def generate_walks(G: dgl.DGLGraph, num_walks: int, walk_length: int):
#     # Runs "num_walks" random walks from each node
#
#     walks = []
#
#     # Please insert your code for Task 1 here
#     idx = torch.randperm(len(G.nodes()))
#     shuffled_nodes = G.nodes()[idx]
#
#     num_nodes = len(shuffled_nodes)
#     batch_size = 32
#
#     with tqdm.tqdm(range(num_walks)) as pbar:
#         for n in range(num_walks):
#             for idx in range(num_nodes // batch_size + 1):
#                 walks.append(dgl.sampling.node2vec_random_walk(G,
#                                                                shuffled_nodes[idx * batch_size:(idx + 1) * batch_size],
#                                                                p=1,
#                                                                q=1,
#                                                                walk_length=walk_length))
#             pbar.update()
#     return torch.cat(walks)

def generate_walks(G: nx.Graph, num_walks: int, walk_length: int):
    # Runs "num_walks" random walks from each node

    walks = []

    # Please insert your code for Task 1 here
    with tqdm.tqdm(range(num_walks)) as pbar:
        for i in range(num_walks):
            permuted_nodes = np.random.permutation(G.nodes())
            for node in permuted_nodes:
                walks.append(random_walk(G, node, walk_length))
            pbar.update()
    return walks


def deepwalk(G: nx.Graph,
             num_walks: int = 2,
             walk_length: int = 20,
             n_dim: int = 128,
             window: int = 8,
             min_count: int = 0,
             sg: int = 1,
             workers: int = 24,
             hs: int = 1,
             epochs: int = 5):
    # Simulates walks and uses the Skipgram model to learn node representations

    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")

    # Please insert your code for Task 2 here
    model = Word2Vec(vector_size=n_dim, window=window, min_count=min_count, sg=sg, workers=workers, hs=hs)

    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=epochs)

    return model


def generate_embeddings(G: nx.Graph, model, n_dims):
    num_nodes = G.number_of_nodes()
    embeddings = np.empty(shape=(num_nodes, n_dims))
    for idx, node in enumerate(G.nodes()):
        embeddings[idx, :] = model.wv[str(node)]

    return embeddings


def train_clf(embeddings, train_dataset, dev_dataset):
    # cos = torch.nn.CosineSimilarity(dim=1)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # u_tc_train = torch.tensor(embeddings[train_dataset['u']]).to(device)
    # v_tc_train = torch.tensor(embeddings[train_dataset['u']]).to(device)
    # u_tc_dev = torch.tensor(embeddings[dev_dataset['u']]).to(device)
    # v_tc_dev = torch.tensor(embeddings[dev_dataset['u']]).to(device)

    X_train = np.concatenate([embeddings[train_dataset['u']], embeddings[train_dataset['v']]], axis=-1)
    # X_train = torch.cat([torch.linalg.norm(u_tc_train - v_tc_train, dim=1).unsqueeze(-1),
    #                      cos(u_tc_train, v_tc_train).unsqueeze(-1)], dim=1).detach().cpu().numpy()

    Y_train = train_dataset['y']

    X_dev = np.concatenate([embeddings[dev_dataset['u']], embeddings[dev_dataset['v']]], axis=-1)
    # X_dev = torch.cat([torch.linalg.norm(u_tc_dev - v_tc_dev, dim=1).unsqueeze(-1),
    #                    cos(u_tc_dev, v_tc_dev).unsqueeze(-1)], dim=1).detach().cpu().numpy()

    Y_dev = dev_dataset['y']

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_dev)
    loss = log_loss(Y_dev, Y_pred)
    score = accuracy_score(Y_dev, Y_pred)
    logger.info(f"Score: {score}, Loss: {loss}")
    return clf, score


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    train_dataset = pickle.load(open(to_absolute_path(cfg.dataset.train_dataset_path), 'rb'))
    dev_dataset = pickle.load(open(to_absolute_path(cfg.dataset.dev_dataset_path), 'rb'))
    test_dataset = pickle.load(open(to_absolute_path(cfg.dataset.test_dataset_path), 'rb'))

    # train_graph = dgl.graph((train_dataset['pos_u'], train_dataset['pos_v']), idtype=torch.int32)
    train_graph = nx.Graph()
    train_graph.add_edges_from(train_dataset['origin_edges'])
    # clf, score = train_clf(np.zeros(shape=(train_graph.number_of_nodes(), 64)), train_dataset, dev_dataset)
    model = deepwalk(train_graph,
                     num_walks=cfg.train.num_walks,
                     walk_length=cfg.train.num_walks,
                     n_dim=cfg.train.n_dim,
                     window=cfg.train.window,
                     min_count=cfg.train.min_count,
                     sg=cfg.train.sg,
                     workers=cfg.train.workers,
                     hs=cfg.train.hs,
                     epochs=cfg.train.epochs)

    embeddings = generate_embeddings(train_graph, model, cfg.train.n_dim)
    clf, score = train_clf(embeddings, train_dataset, dev_dataset)
    with open(cfg.io.embedding_save_path, 'wb') as f:
        pickle.dump({'deep_walk_embeddings': embeddings}, f)
        logger.info(f"saving embeddings to {cfg.io.embedding_save_path}")
    print('finish')


if __name__ == '__main__':
    main()
