from src.utlis import NetworkDataset, generate_submission
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import csv
import numpy as np


def calculate_score(clf, X_dev, y_dev):
    y_pred = clf.predict_proba(X_dev)
    y_pred = y_pred[:, 1]
    loss_val = metrics.log_loss(y_dev, y_pred)

    y_pred = (y_pred > 0.5).astype(int)
    f1_score = metrics.f1_score(y_dev, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    acc = metrics.accuracy_score(y_dev, y_pred)
    print(f"Loss: {loss_val}, Accuracy: {acc}, F1-score: {f1_score}")
    return loss_val, acc, f1_score


if __name__ == '__main__':
    train_set = NetworkDataset('../../data/converted/nullptr_train.pkl')
    dev_set = NetworkDataset('../../data/converted/nullptr_dev.pkl')
    test_set = NetworkDataset('../../data/converted/nullptr_test.pkl')

    # Read test data. Each sample is a pair of nodes
    X_train = train_set.cache[:, :train_set.feature_dim]
    y_train = train_set.cache[:, train_set.feature_dim:].ravel()

    X_dev = dev_set.cache[:, :dev_set.feature_dim]
    y_dev = dev_set.cache[:, dev_set.feature_dim:].ravel()

    # Create the test matrix. Use the same 4 features as above
    X_test = test_set.cache[:, :test_set.feature_dim]
    print('Size of training matrix:', X_test.shape)

    # Use logistic regression to predict if two nodes are linked by an edge
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    calculate_score(clf, X_dev, y_dev)

    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]
    generate_submission("./", y_pred)
