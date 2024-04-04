import numpy as np
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from functools import partial

logistic_regression = LogisticRegression
mlp_classifier = partial(MLPClassifier, early_stopping=True, l2_regularization=0.1)
lda = LinearDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis
hist_gradboost = partial(
    HistGradientBoostingClassifier, early_stopping=True, l2_regularization=1
)
adaboost = AdaBoostClassifier


def get_clf(clf_name, **kwargs):
    if clf_name == "logistic_regression":
        return logistic_regression(**kwargs)
    elif clf_name == "MLP":
        return mlp_classifier(**kwargs)
    elif clf_name == "LDA":
        return lda(**kwargs)
    elif clf_name == "QDA":
        return qda(**kwargs)
    elif clf_name == "HistGrad":
        return hist_gradboost(**kwargs)
    elif clf_name == "AdaBoost":
        return adaboost(**kwargs)
    else:
        raise ValueError


def clf_twostage(
    clf_name, X_train, X_test, y_train, y_test, X_whole, y_whole, pca_components=None
):

    y_train_step1 = np.copy(y_train)
    y_train_step1[y_train != 0] = 1

    y_test_step1 = np.copy(y_test)
    y_test_step1[y_test != 0] = 1

    steps = [("scaler", StandardScaler()), ("clf", get_clf(clf_name))]
    if pca_components is not None:
        steps.insert(1, PCA(n_components=pca_components))
    clf_step1 = make_pipeline(*steps)
    clf_step1.fit(X_train, y_train_step1)
    acc_step1 = clf_step1.score(X_test, y_test_step1)
    print("Accuracy Step 1", acc_step1)

    y_train_step2 = np.copy(y_train)
    X_train_step2 = np.copy(X_train)

    silence_index = np.where(y_train == 0)[0]
    X_train_step2 = np.delete(X_train_step2, silence_index, axis=0)
    y_train_step2 = np.delete(y_train_step2, silence_index, axis=0)
    y_train_step2 = y_train_step2 - 1

    steps = [("scaler", StandardScaler()), ("clf", get_clf(clf_name))]
    if pca_components is not None:
        steps.insert(1, PCA(n_components=pca_components))
    clf_step2 = make_pipeline(*steps)
    clf_step2.fit(X_train_step2, y_train_step2)

    X_test_step2 = np.copy(X_test)
    y_test_step2 = np.copy(y_test)
    silence_index = np.where(y_test == 0)[0]
    X_test_step2 = np.delete(X_test_step2, silence_index, axis=0)
    y_test_step2 = np.delete(y_test_step2, silence_index, axis=0)
    y_test_step2 = y_test_step2 - 1

    acc_step2 = clf_step2.score(X_test_step2, y_test_step2)
    print("Accuracy Step 2", acc_step2)

    # ConfusionMatrixDisplay.from_predictions(
    #     clf_step2.predict(X_test_step2),
    #     y_test_step2,
    #     normalize="all",
    # )
    # plt.show()

    step1 = clf_step1.predict_proba(X_whole)
    step2 = clf_step2.predict_proba(X_whole)

    y_prb_train = twostage_predictor(X_train, step1, step2)
    y_prb_test = twostage_predictor(X_test, step1, step2)
    y_prb_whole = twostage_predictor(X_whole, step1, step2)

    return y_prb_train, y_prb_test, y_prb_whole


def twostage_predictor(X, step1, step2):
    y_prb = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        if np.argmax(step1[i]) == 0:
            y_prb[i] = 0
        else:
            y_prb[i] = np.argmax(step2[i]) + 1
    prb = np.zeros((y_prb.size, y_prb.max() + 1), dtype=int)
    prb[np.arange(y_prb.size), y_prb] = 1
    return y_prb


def clf_onestage(
    clf_name, X_train, X_test, y_train, y_test, X_whole, y_whole, pca_components=None
):
    steps = [StandardScaler(), get_clf(clf_name)]
    if pca_components is not None:
        steps.insert(1, PCA(n_components=pca_components))
    clf = make_pipeline(*steps)
    clf.fit(X_train, y_train)
    y_prb_train = clf.predict_proba(X_train)
    y_prb_test = clf.predict_proba(X_test)
    y_prb_whole = clf.predict_proba(X_whole)
    return y_prb_train, y_prb_test, y_prb_whole
