import networkx as nx

from src.utils.fitter import fit_lr_classifier, infer_lr_classifier, calculate_score
import pickle
import numpy as np
from src.utils.submmision import generate_submission
from src.utils.io import dump_features
import os
import itertools
import dgl
import torch
from typing import Tuple
from src.models import GraphSAGEBundled
import torch.nn.functional as torch_f
import tqdm
import torch.nn as nn
from typing import List
import math
import dgl.function as fn


class Voter:
    def __init__(self, estimators: list, weight=None):
        self.estimators_lookup = {name: idx for idx, (name, _) in enumerate(estimators)}
        self.estimators = [item for name, item in estimators]
        if weight is not None:
            self.weight = weight / max(weight)
        else:
            self.weight = None

    def fit(self, x, y, *args, **kwargs):
        for name, data in x.keys():
            self.estimators[self.estimators_lookup[name]].fit(data, y, *args, **kwargs)
            print(f"fitting: {name}")

    def predict_proba(self, x):
        predictions = [self.estimators[self.estimators_lookup[name]].predict_proba(x[name]) for name in x.keys()]
        summary = 0

        for idx, partial_result in enumerate(predictions):
            if self.weight is not None:
                summary += partial_result * self.weight[idx]
            else:
                summary += 1 / len(self.estimators) * partial_result

        return summary


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')



    # GraphSAGE on cora_like passages, rare word
    cora_hidden_state = pickle.load(open(cora_hidden_state_path, 'rb'))
    cora_hidden_state = cora_hidden_state.detach().cpu().numpy()

    # get_feature(0, 1,
    #             whole_dataset['authors'],
    #             whole_dataset['authors_index_mapping'],
    #             authors_graph_nx,
    #             authors_graph_weights,
    #             ctx['page_rank'])

    X_train_1 = get_graph_feature_array(train_dataset['u'], train_dataset['v'], graphsage_state['h'], cora_hidden_state, whole_dataset['authors'], whole_dataset['authors_index_mapping'])
    X_dev_1 = get_graph_feature_array(dev_dataset['u'], dev_dataset['v'], graphsage_state['h'], cora_hidden_state, whole_dataset['authors'], whole_dataset['authors_index_mapping'])
    X_test_1 = get_graph_feature_array(test_dataset['u'], test_dataset['v'], graphsage_state['h'], cora_hidden_state, whole_dataset['authors'], whole_dataset['authors_index_mapping'])

    whole_authors_graph = ctx['whole_graph']
    whole_authors_graph.edata['weight'] = torch.tensor(whole_dataset['authors_graph_weights']).to(whole_authors_graph.device)
    authors_graph_nx = whole_authors_graph.cpu().to_networkx()
    X_train_2 = get_feature_array(train_dataset['u'],
                                  train_dataset['v'],
                                  11,
                                  whole_dataset['authors'],
                                  whole_dataset['authors_index_mapping'],
                                  authors_graph_nx,
                                  authors_graph_weights,
                                  ctx['page_rank'])  # Loss: 0.2263460069674607, Accuracy: 0.9180460733271976, F1-score: 0.9205102126922513, using 0,4,7,8,9,10
    X_dev_2 = get_feature_array(dev_dataset['u'],
                                dev_dataset['v'],
                                11,
                                whole_dataset['authors'],
                                whole_dataset['authors_index_mapping'],
                                authors_graph_nx,
                                authors_graph_weights,
                                ctx['page_rank'])
    X_test_2 = get_feature_array(test_dataset['u'],
                                 test_dataset['v'],
                                 11,
                                 whole_dataset['authors'],
                                 whole_dataset['authors_index_mapping'],
                                 authors_graph_nx,
                                 authors_graph_weights,
                                 ctx['page_rank'])

    Y_train = train_dataset['y']
    Y_dev = dev_dataset['y']

    clf1 = fit_lr_classifier(X_train_1, Y_train, X_dev_1, Y_dev, max_iter=200)
    clf2 = fit_lr_classifier(X_train_2[:[0, 4, 7, 8, 9, 10]], Y_train[:[0, 4, 7, 8, 9, 10]], X_dev_2, Y_dev, max_iter=200)

    voter = Voter(estimators=[('lr1', clf1), ('lr2', clf2)])
    calculate_score(voter, {'lr1': X_dev_1, 'lr2': X_dev_2[:, [0, 4, 7, 8, 9, 10]]}, Y_dev)

    scores = infer_lr_classifier(voter, {'lr1': X_test_1, 'lr2': X_test_2[:, [0, 4, 7, 8, 9, 10]]})
    generate_submission('./outputs', scores, "graph_sage_plus_author_network")
    with open('./graphsage_state.pkl', 'wb') as f:
        pickle.dump(graphsage_state, f)
    # from sklearn.ensemble import VotingRegressor
    # voting_clf = VotingRegressor(
    #     estimators=[('lr1', clf1), ('lr2', clf2)],
    # )
    # voting_clf.predict(X_dev_1, X_dev_2)
    # model, embeddings = train_graphsage_essay(ctx, epochs=200, device=device)

    #
    #
    # clf = fit_lr_classifier(X_train, Y_train, 1000)
    # Y_pred = infer_lr_classifier(clf, X_dev)
    # calculate_score(clf, X_dev, Y_dev)
    #
    # clf = fit_lr_classifier(X_whole, Y_whole, 1000)
    # scores = infer_lr_classifier(clf, X_test)
    # generate_submission('./', scores, "deep_walk")
    #
    # dump_features('./', X_whole, Y_whole, X_train, Y_train, X_dev, Y_dev, X_test, 'deep_walk')
