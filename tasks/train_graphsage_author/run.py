import itertools
import pickle
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import tqdm

from src.models import GraphSAGEBundled
from src.utils import split_graph
from src.utils.io import check_md5


def compute_train_loss(pos_score: torch.Tensor, neg_score: torch.Tensor):
    device = pos_score.device
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).to(device)
    loss = torch_f.binary_cross_entropy(scores, labels)
    return loss


def train_graphsage(graph_ctx,
                    model: nn.Module = None,
                    authors_embedding: nn.Module = None,
                    embedding_dims: int = 128,
                    hidden_dims: int = 32,
                    output_dims: int = 32,
                    dropout: float = 0.1,
                    epochs: int = 2,
                    lr: float = 1e-3,
                    device: torch.device = torch.device('cpu')):
    num_nodes = graph_ctx['num_nodes']
    # train_graph = graph_ctx['train_graph']

    model = GraphSAGEBundled(embedding_dims,
                             hidden_dims,
                             output_dims,
                             "dot",
                             "mean",
                             dropout=dropout) if model is None else model
    model = model.to(device)
    authors_embedding = nn.Embedding(num_nodes, embedding_dims) if authors_embedding is None else authors_embedding
    authors_embedding = authors_embedding.to(device)

    optimizer = torch.optim.Adam(itertools.chain(authors_embedding.parameters(), model.parameters()),
                                 lr=lr)

    for epoch_idx in range(epochs):
        model.train()
        optimizer.zero_grad()

        features = authors_embedding(torch.arange(0, num_nodes).to(device))
        h = model(graph_ctx['whole_graph'], features)
        train_pos_score = model.predictor(graph_ctx['whole_graph'], h)  # Using the entire graph for training
        train_neg_score = model.predictor(graph_ctx['whole_graph'], h)
        train_loss = compute_train_loss(train_pos_score, train_neg_score)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            dev_pos_score = model.predictor(graph_ctx['dev_pos_graph'], h)
            dev_neg_score = model.predictor(graph_ctx['dev_neg_graph'], h)
            dev_loss = compute_train_loss(dev_pos_score, dev_neg_score)
            print(f"epoch={epoch_idx}, train_loss={train_loss.detach().cpu().numpy()}, dev_loss={dev_loss.detach().cpu().numpy()}")

    return model, authors_embedding


def get_graph_feature_array(u_array,
                            v_array,
                            author_hidden_state: np.ndarray,
                            authors: List[List[str]],
                            authors_index_mapping: Dict[str, int]):
    num_features: int = author_hidden_state.shape[1]
    res = np.zeros(shape=(len(u_array), num_features))

    with tqdm.tqdm(range(len(res))) as pbar:
        for idx in range(res.shape[0]):
            u, v = u_array[idx], v_array[idx]
            u_main_author, v_main_author = authors_index_mapping[authors[u][0]], authors_index_mapping[authors[v][0]]
            res[idx] = (author_hidden_state[u_main_author] * author_hidden_state[v_main_author]) / \
                       (1e-8 +
                        np.linalg.norm(author_hidden_state[u_main_author]) *
                        np.linalg.norm(author_hidden_state[v_main_author]))

            # if idx % 1e3 == 0 and idx != 0: pbar.update(1e3)
            pbar.update()
    return res


from src.utils.fitter import fit_lr_classifier, infer_lr_classifier

if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    # Load uv
    uv_list_path = '../../data/neo_converted/uv_list.pkl'
    print(check_md5(uv_list_path))
    uv_dataset = pickle.load(open(uv_list_path, 'rb'))

    # Load dataset
    whole_dataset_path = '../../data/converted/nullptr_graph_conv_whole.pkl'
    # train_dataset_path = '../../data/converted/nullptr_graph_conv_train.pkl'
    # dev_dataset_path = '../../data/converted/nullptr_graph_conv_dev.pkl'
    # test_dataset_path = '../../data/converted/nullptr_graph_conv_test.pkl'

    whole_dataset = pickle.load(open(whole_dataset_path, 'rb'))
    # train_dataset = pickle.load(open(train_dataset_path, 'rb'))
    # dev_dataset = pickle.load(open(dev_dataset_path, 'rb'))
    # test_dataset = pickle.load(open(test_dataset_path, 'rb'))

    authors_graph_edges = whole_dataset['authors_graph_edges']
    authors_graph_weights = whole_dataset['authors_graph_weights']
    authors_mapping = whole_dataset['authors_index_mapping']

    # GraphSAGE on authors network
    ctx = split_graph(authors_graph_edges, device=device)
    model, embeddings = train_graphsage(ctx, epochs=2000, device=device)
    graphsage_state = {
        'h': model.hidden_state.detach().cpu().numpy(),
        'embeddings': embeddings.state_dict(),
        'model': model.state_dict()
    }
    with open('./graphsage_author_state.pkl', 'wb') as f:
        pickle.dump(graphsage_state, f)

    # get_feature(0, 1,
    #             whole_dataset['authors'],
    #             whole_dataset['authors_index_mapping'],
    #             authors_graph_nx,
    #             authors_graph_weights,
    #             ctx['page_rank'])

    X_train = get_graph_feature_array(uv_dataset['train_u'], uv_dataset['train_v'], graphsage_state['h'], whole_dataset['authors'], whole_dataset['authors_index_mapping'])
    X_dev = get_graph_feature_array(uv_dataset['dev_u'], uv_dataset['dev_v'], graphsage_state['h'], whole_dataset['authors'], whole_dataset['authors_index_mapping'])
    X_test = get_graph_feature_array(uv_dataset['test_u'], uv_dataset['test_v'], graphsage_state['h'], whole_dataset['authors'], whole_dataset['authors_index_mapping'])

    whole_authors_graph = ctx['whole_graph']
    whole_authors_graph.edata['weight'] = torch.tensor(whole_dataset['authors_graph_weights']).to(whole_authors_graph.device)
    authors_graph_nx = whole_authors_graph.cpu().to_networkx()

    Y_train = uv_dataset['train_y']
    Y_dev = uv_dataset['dev_y']

    features = {
        '__text__': 'graphsage_author_features',
        'uv_chksum': check_md5(uv_list_path),
        'X_train': X_train,
        'Y_train': uv_dataset['train_y'],
        'X_dev': X_dev,
        'Y_dev': uv_dataset['dev_y'],
        'X_test': X_test,
    }

    clf = fit_lr_classifier(X_train, Y_train, X_dev, Y_dev, max_iter=200)

    scores = infer_lr_classifier(clf, X_test)
    # generate_submission('./outputs', scores, "graphsage_author")

    with open('./graphsage_author_features.pkl', 'wb') as f:
        pickle.dump(features, f)
