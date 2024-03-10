from os.path import join
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def hist(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    plt.bar(unique_values, counts / np.sum(counts))


def get_center(lbl, melSpec, n_clusters):
    center = np.zeros((n_clusters, melSpec.shape[1]))
    for i in range(n_clusters):
        center[i] = np.mean(melSpec[lbl == i], axis=0)
    return center


if __name__ == "__main__":
    num_classes = 20
    feature_folder = "../Dataset_Sentence"  #  "../Dataset_Word"
    path_input = feature_folder  # join(feature_folder, "features")
    participant = "p07_ses1_sentences"  #   "sub-06"
    feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
    cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))

    X_train, X_test, y_train, y_test = train_test_split(
        feat, cluster, test_size=0.2, random_state=0
    )

    clf = HistGradientBoostingClassifier(
        max_iter=1000,
        max_leaf_nodes=100,
        random_state=0,
        verbose=1,
        early_stopping=True,
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
