from src.utils import NetworkDatasetEdge, generate_submission
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import csv
import numpy as np
import torch.nn as nn
import torch

from mlpfitter import BasicBlock, fit_mlp_classifier, infer_mlp_classifier


def calculate_score(clf, X_dev, y_dev):
    y_pred = infer_mlp_classifier(clf, X_dev).detach().cpu().numpy()
    loss_val = metrics.log_loss(y_dev, y_pred)

    y_pred = (y_pred > 0.5).astype(int)
    f1_score = metrics.f1_score(y_dev, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    acc = metrics.accuracy_score(y_dev, y_pred)
    print(f"Loss: {loss_val}, Accuracy: {acc}, F1-score: {f1_score}")
    return loss_val, acc, f1_score


if __name__ == '__main__':
    device = torch.device('cuda:2')
    train_set = NetworkDatasetEdge('../../data/neo_converted/nullptr_whole.pkl')
    dev_set = NetworkDatasetEdge('../../data/neo_converted/nullptr_dev.pkl')
    test_set = NetworkDatasetEdge('../../data/neo_converted/nullptr_test.pkl')

    # Read test data. Each sample is a pair of nodes
    X_train = train_set.edge_features[:, :train_set.edge_feature_dim]
    y_train = train_set.y.ravel()

    X_dev = dev_set.edge_features[:, :dev_set.edge_feature_dim]
    y_dev = dev_set.y.ravel()

    # Create the test matrix. Use the same 4 features as above
    X_test = test_set.edge_features[:, :test_set.edge_feature_dim]
    print('Size of training matrix:', X_test.shape)

    # Use logistic regression to predict if two nodes are linked by an edge
    clf = nn.Sequential(
        BasicBlock(5, 64),
        BasicBlock(64, 1),
        nn.LogSigmoid()
    )
    clf = fit_mlp_classifier(clf,
                             X_train,
                             y_train,
                             batch_size=128,
                             epochs=5,
                             lr=0.01,
                             device=device)

    calculate_score(clf, X_dev, y_dev)

    y_pred = infer_mlp_classifier(clf, X_test).detach().cpu().numpy()
    generate_submission("./", y_pred, 'mlp')
