import pickle

import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression

from src.utils import NetworkDatasetEdge, generate_submission


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def calculate_score(clf, X_dev, y_dev):
    y_pred = clf.predict_proba(X_dev)
    # y_pred = clf.predict(X_dev)
    y_pred = y_pred[:, 1]
    loss_val = metrics.log_loss(y_dev, y_pred)

    y_pred = (y_pred > 0.5).astype(int)
    f1_score = metrics.f1_score(y_dev, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    acc = metrics.accuracy_score(y_dev, y_pred)
    print(f"Loss: {loss_val}, Accuracy: {acc}, F1-score: {f1_score}")
    return loss_val, acc, f1_score


def calculate_score_svm(clf, X_dev, y_dev):
    y_pred = clf.predict(X_dev)
    loss_val = metrics.log_loss(y_dev, y_pred)

    y_pred = (y_pred > 0.5).astype(int)
    f1_score = metrics.f1_score(y_dev, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    acc = metrics.accuracy_score(y_dev, y_pred)
    print(f"Loss: {loss_val}, Accuracy: {acc}, F1-score: {f1_score}")
    return loss_val, acc, f1_score


if __name__ == '__main__':
    train_set = NetworkDatasetEdge('../../data/neo_converted/nullptr_no_feature_train.pkl')
    dev_set = NetworkDatasetEdge('../../data/neo_converted/nullptr_no_feature_dev.pkl')
    test_set = NetworkDatasetEdge('../../data/neo_converted/nullptr_no_feature_test.pkl')

    # train_set.dump_shortest_path()
    # dev_set.dump_shortest_path()
    # test_set.dump_shortest_path()

    # # Read test data. Each sample is a pair of nodes
    X_train = train_set.features
    X_train[:, 10] = sigmoid(X_train[:, 10])
    y_train = train_set.y.ravel()
    #
    X_dev = dev_set.features
    X_dev[:, 10] = sigmoid(X_dev[:, 10])
    y_dev = dev_set.y.ravel()

    # Create the test matrix. Use the same 4 features as above
    X_test = test_set.features
    X_test[:, 10] = sigmoid(X_test[:, 10])
    print('Size of training matrix:', X_test.shape)

    # Use logistic regression to predict if two nodes are linked by an edge
    clf = LogisticRegression(
        solver='lbfgs',
        tol=1e-5,
        max_iter=1000,
        n_jobs=12,
        verbose=1,
    )

    # Use
    clf.fit(X_train, y_train)

    calculate_score(clf, X_dev, y_dev)
    # calculate_score_svm(clf, X_dev, y_dev)

    y_pred = clf.predict_proba(X_test)
    # y_pred = clf.predict(X_test)
    y_pred = y_pred[:, 1]
    generate_submission("./", y_pred)

    features = {
        '__text__': 'baseline-enhanced',
        'X_train': X_train,
        'Y_train': y_train,
        'X_dev': X_dev,
        'Y_dev': y_dev,
        'X_test': X_test,
        'train_u': train_set.u,
        'train_v': train_set.v,
        'dev_u': dev_set.u,
        'dev_v': dev_set.v,
    }
    with open('baseline-enhanced.pkl', 'wb') as f:
        pickle.dump(features, f)
