from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import numpy as np
from os.path import join
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from reconstruction_minimal import createAudio
import scipy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import warnings

# warnings.filterwarnings("ignore")


def get_clf(clf_name):
    if clf_name == "logistic_regression":
        return LogisticRegression()
    elif clf_name == "MLP":
        return MLPClassifier(early_stopping=True)
    elif clf_name == "LDA":
        return LinearDiscriminantAnalysis()
    elif clf_name == "QDA":
        return QuadraticDiscriminantAnalysis()
    elif clf_name == "HistGrad":
        return HistGradientBoostingClassifier()
    elif clf_name == "AdaBoost":
        return AdaBoostClassifier()
    elif clf_name == "SVC":
        return SVC(probability=True)
    else:
        raise ValueError


def clf_twostage(clf_name, feat, cluster, train, test):

    X_train, y_train = feat[train], cluster[train]
    X_test, y_test = feat[test], cluster[test]

    y_train_step1 = np.copy(y_train)
    y_train_step1[y_train != 0] = 1

    y_test_step1 = np.copy(y_test)
    y_test_step1[y_test != 0] = 1

    clf_step1 = make_pipeline(StandardScaler(), get_clf(clf_name))
    clf_step1.fit(X_train, y_train_step1)
    acc_step1 = clf_step1.score(X_test, y_test_step1)
    print("Accuracy Step 1", acc_step1)

    y_train_step2 = np.copy(y_train)
    X_train_step2 = np.copy(X_train)

    silence_index = np.where(y_train == 0)[0]
    X_train_step2 = np.delete(X_train_step2, silence_index, axis=0)
    y_train_step2 = np.delete(y_train_step2, silence_index, axis=0)
    y_train_step2 = y_train_step2 - 1

    clf_step2 = make_pipeline(StandardScaler(), get_clf(clf_name))
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

    step1 = clf_step1.predict_proba(feat)
    step2 = clf_step2.predict_proba(feat)

    pred = np.zeros(feat.shape[0], dtype=int)
    for i in range(feat.shape[0]):
        if np.argmax(step1[i]) == 0:
            pred[i] = 0
        else:
            pred[i] = np.argmax(step2[i]) + 1
    prb = np.zeros((pred.size, pred.max() + 1), dtype=int)
    prb[np.arange(pred.size), pred] = 1
    return prb


def clf_onestage(clf_name, feat, cluster, train, test):
    clf = make_pipeline(StandardScaler(), get_clf(clf_name))
    clf.fit(feat[train], cluster[train])
    prb0 = clf.predict_proba(feat)
    return prb0


def main():
    num_classes = 2
    nfolds = 10
    path_input = "../Dataset_Word"  # "../Dataset_Sentence"
    participant = "sub-06"  #    "p07_ses1_sentences"
    audiosr = 16000
    save_reconstruction = False

    clf_name = "QDA"
    method = "onestage"
    clustering_method = "kmeans"

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))
    cluster = np.load(
        join(
            clustering_method,
            f"{participant}_spec_cluster_{num_classes}_kfold_{nfolds}.npy",
        )
    )
    melSpec_centers = np.load(
        join(
            clustering_method,
            f"{participant}_spec_cluster_{num_classes}_kfold_{nfolds}_centers.npy",
        )
    )

    kf = KFold(nfolds, shuffle=False)

    prb = np.zeros((nfolds, melSpec.shape[0], num_classes), dtype=int)

    for k, (train, test) in enumerate(kf.split(melSpec)):
        if np.unique(cluster[k, train]).size < np.unique(cluster[k]).size:
            print("K", k, "Skipping due to missing classes")
            continue
        if method == "onestage":
            prb0 = clf_onestage(clf_name, feat, cluster[k], train, test)
        elif method == "twostage":
            prb0 = clf_twostage(clf_name, feat, cluster[k], train, test)
        pred0 = np.argmax(prb0, 1)
        acc_test = (pred0[test] == cluster[k, test]).mean()
        acc_train = (pred0[train] == cluster[k, train]).mean()
        cluster_hist = np.unique(cluster[k, test], return_counts=True)
        ConfusionMatrixDisplay.from_predictions(
            cluster[k, test],
            pred0[test],
            normalize="all",
        )
        plt.gcf().set_size_inches(8, 8)
        plt.title(
            f"{participant}, {num_classes} classes, {clf_name}, {method} method\nkfold iteration: {k} Accuracy: {acc_test:.3%}"
        )
        plt.savefig(
            join(
                clf_name,
                f"{participant}_cluster_{clustering_method}_{num_classes}_{clf_name}_{method}_k_{k}_confusion.png",
            )
        )
        print(
            k,
            f"train acc {acc_train:.3} test acc {acc_test:.3} max histogram {np.max(cluster_hist[1])/np.sum(cluster_hist[1]):.3}",
        )
        prb[k, :, : prb0.shape[1]] = prb0
    np.save(
        join(
            clf_name,
            f"{participant}_cluster_{clustering_method}_{num_classes}_{clf_name}_{method}_prb",
        ),
        prb,
    )
    if save_reconstruction:
        k = 0
        melSpec_pred = np.stack(
            tuple(melSpec_centers[k][x] for x in np.argmax(prb[k], 1)), axis=0
        )
        center_audio = createAudio(melSpec_pred, audiosr)
        scipy.io.wavfile.write(
            join(
                clf_name,
                f"{participant}_cluster_{clustering_method}_{num_classes}_{clf_name}_{method}_k_{k}_reconstructed.wav",
            ),
            int(audiosr),
            center_audio,
        )


if __name__ == "__main__":
    main()
