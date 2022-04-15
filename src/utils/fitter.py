from src.utils.driver import NetworkDatasetEdge
from src.utils.submmision import generate_submission
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics


def calculate_score(clf, X_dev, Y_dev):
    Y_pred = clf.predict_proba(X_dev)
    Y_pred = Y_pred[:, 1]
    loss_val = metrics.log_loss(Y_dev, Y_pred)

    Y_pred = (Y_pred > 0.5).astype(int)
    f1_score = metrics.f1_score(Y_dev, Y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    acc = metrics.accuracy_score(Y_dev, Y_pred)
    print(f"Loss: {loss_val}, Accuracy: {acc}, F1-score: {f1_score}")
    return loss_val, acc, f1_score


def calculate_score_raw(Y_dev, Y_pred):
    loss_val = metrics.log_loss(Y_dev, Y_pred)
    Y_pred = (Y_pred > 0.5).astype(int)
    f1_score = metrics.f1_score(Y_dev, Y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    acc = metrics.accuracy_score(Y_dev, Y_pred)
    print(f"Loss: {loss_val}, Accuracy: {acc}, F1-score: {f1_score}")
    return loss_val, acc, f1_score


def fit_lr_classifier(X_train,
                      Y_train,
                      X_dev=None,
                      Y_dev=None,
                      *args,
                      **kwargs):
    clf = LogisticRegression(*args, **kwargs)
    clf.fit(X_train, Y_train)

    if X_dev is not None and Y_dev is not None:
        calculate_score(clf, X_dev, Y_dev)

    return clf


def infer_lr_classifier(clf,
                        X_test):
    pred = clf.predict_proba(X_test)
    return pred[:, 1]
