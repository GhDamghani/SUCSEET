from os.path import join
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    num_classes = 20
    feature_folder = "../Dataset_Word"  # "../Kmeans of sounds"
    path_input = feature_folder  # join(feature_folder, "features")
    participant = "sub-06"  #  "p07_ses1_sentences"
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

    clf_step1 = LinearDiscriminantAnalysis()
    clf_step1.fit(X_train, y_train_step1)

    acc_step1 = clf_step1.score(X_test, y_test_step1)
    print(acc_step1)

    y_train_step2 = np.copy(y_train)
    X_train_step2 = np.copy(X_train)

    silence_index = np.where(y_train == 0)[0]
    np.delete(X_train_step2, silence_index, axis=0)
    np.delete(y_train_step2, silence_index, axis=0)
    y_train_step2 = y_train_step2 - 1

    clf_step2 = LinearDiscriminantAnalysis()
    clf_step2.fit(X_train_step2, y_train_step2)

    X_test_step2 = np.copy(X_test)
    y_test_step2 = np.copy(y_test)
    silence_index = np.where(y_test == 0)[0]
    np.delete(X_test_step2, silence_index, axis=0)
    np.delete(y_test_step2, silence_index, axis=0)
    y_test_step2 = y_test_step2 - 1

    acc_step2 = clf_step2.score(X_test_step2, y_test_step2)
    print(acc_step2)

    def two_step_classifier(X):
        X = np.expand_dims(X, 0)
        step1 = clf_step1.predict(X)
        if step1 == 0:
            return 0
        step2 = clf_step2.predict(X).item()
        return step2 + 1

    y_test_pred = np.apply_along_axis(two_step_classifier, 1, X_test)
    acc = np.sum(y_test_pred == y_test) / len(y_test)
    print(acc)
