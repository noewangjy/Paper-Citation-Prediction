import pickle

import numpy as np

from src.utils.fitter import fit_lr_classifier, infer_lr_classifier, calculate_score
from src.utils.io import dump_features
from src.utils.submission import generate_submission

if __name__ == '__main__':
    whole_dataset_path = '../../data/converted/nullptr_graph_conv_whole.pkl'
    train_dataset_path = '../../data/converted/nullptr_graph_conv_train.pkl'
    dev_dataset_path = '../../data/converted/nullptr_graph_conv_dev.pkl'
    test_dataset_path = '../../data/converted/nullptr_graph_conv_test.pkl'

    embeddings_pkl_path = '/disk1/home/yutong/DataChallenge-graph/tasks/run_deep_walk/outputs/2022-04-11/19-23-58/embeddings.pkl'
    embeddings = pickle.load(open(embeddings_pkl_path, 'rb'))['deep_walk_embeddings']

    whole_dataset = pickle.load(open(whole_dataset_path, 'rb'))
    train_dataset = pickle.load(open(train_dataset_path, 'rb'))
    dev_dataset = pickle.load(open(dev_dataset_path, 'rb'))
    test_dataset = pickle.load(open(test_dataset_path, 'rb'))

    X_whole = np.concatenate([embeddings[whole_dataset['u']], embeddings[whole_dataset['v']]], axis=1)
    Y_whole = whole_dataset['y']

    X_train = np.concatenate([embeddings[train_dataset['u']], embeddings[train_dataset['v']]], axis=1)
    Y_train = train_dataset['y']

    X_dev = np.concatenate([embeddings[dev_dataset['u']], embeddings[dev_dataset['v']]], axis=1)
    Y_dev = dev_dataset['y']

    X_test = np.concatenate([embeddings[test_dataset['u']], embeddings[test_dataset['v']]], axis=1)

    clf = fit_lr_classifier(X_train, Y_train, max_iter=1000)
    Y_pred = infer_lr_classifier(clf, X_dev)
    calculate_score(clf, X_dev, Y_dev)

    clf = fit_lr_classifier(X_whole, Y_whole, max_iter=1000)
    scores = infer_lr_classifier(clf, X_test)
    generate_submission('../run_new_baseline/', scores, "deep_walk")

    dump_features('../run_new_baseline/', X_whole, Y_whole, X_train, Y_train, X_dev, Y_dev, X_test, 'deep_walk')
