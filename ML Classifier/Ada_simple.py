from os.path import join
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


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

    speech_index = np.where(y_train != 0)[0]

    y_train_step1 = np.copy(y_train)
    y_train_step1[speech_index] = 1

    y_test_step1 = np.copy(y_test)
    speech_index = np.where(y_test != 0)[0]
    y_test_step1[speech_index] = 1

    shallow_tree = DecisionTreeClassifier(max_depth=3)

    clf = AdaBoostClassifier(
        estimator=shallow_tree,
        n_estimators=100,
        algorithm="SAMME",
        learning_rate=1e-1,
        random_state=0,
    )
    clf.fit(X_train, y_train_step1)

    acc = clf.score(X_test, y_test_step1)
    print(acc)
