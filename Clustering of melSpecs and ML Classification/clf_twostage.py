from os.path import join
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from reconstruction_minimal import createAudio


def hist(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    plt.bar(unique_values, counts / np.sum(counts))


def reconstructAudio(lbl, audiosr, centers):
    center_melSpec = np.stack(tuple(centers[x] for x in lbl), axis=0)
    center_audio = createAudio(center_melSpec, audiosr)
    return center_audio


def get_center(lbl, melSpec, n_clusters):
    center = np.zeros((n_clusters, melSpec.shape[1]))
    for i in range(n_clusters):
        center[i] = np.mean(melSpec[lbl == i], axis=0)
    return center


if __name__ == "__main__":
    save_classification = True
    save_reconstruction = True
    save_confusion_matrix = True
    num_classes = 20
    feature_folder = "../Dataset_Word"  #  "../Dataset_Sentence"
    path_input = feature_folder  # join(feature_folder, "features")
    participant = "sub-06"  #   "p07_ses1_sentences"

    feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))

    clf_name = "QDA"
    CLF = {"QDA": QuadraticDiscriminantAnalysis}
    parameters = {"QDA": {}}

    nfolds = 10
    kf = KFold(nfolds, shuffle=False)

    prb = np.zeros((cluster.shape[0], num_classes), dtype=int)

    for k, (train, test) in enumerate(kf.split(feat)):
        X_train, y_train = feat[train], cluster[train]
        X_test, y_test = feat[test], cluster[test]

        y_train_step1 = np.copy(y_train)
        y_train_step1[y_train != 0] = 1

        y_test_step1 = np.copy(y_test)
        y_test_step1[y_test != 0] = 1

        clf_step1 = CLF[clf_name](**parameters[clf_name])
        clf_step1.fit(X_train, y_train_step1)
        acc_step1 = clf_step1.score(X_test, y_test_step1)
        print(k, "Accuracy Step 1", acc_step1)

        y_train_step2 = np.copy(y_train)
        X_train_step2 = np.copy(X_train)

        silence_index = np.where(y_train == 0)[0]
        np.delete(X_train_step2, silence_index, axis=0)
        np.delete(y_train_step2, silence_index, axis=0)
        y_train_step2 = y_train_step2 - 1

        clf_step2 = CLF[clf_name](**parameters[clf_name])
        clf_step2.fit(X_train_step2, y_train_step2)

        X_test_step2 = np.copy(X_test)
        y_test_step2 = np.copy(y_test)
        silence_index = np.where(y_test == 0)[0]
        np.delete(X_test_step2, silence_index, axis=0)
        np.delete(y_test_step2, silence_index, axis=0)
        y_test_step2 = y_test_step2 - 1

        acc_step2 = clf_step2.score(X_test_step2, y_test_step2)
        print(k, "Accuracy Step 2", acc_step2)

        def two_step_classifier(X):
            X = np.expand_dims(X, 0)
            step1 = clf_step1.predict_proba(X)
            if np.argmax(step1) == 0:
                return np.concatenate(
                    (step1[:, 0:1], np.zeros((1, num_classes - 1))), axis=1
                )
            step2 = clf_step2.predict_proba(X)
            return np.concatenate((np.zeros((1, 1)), step2[:, 1:]), axis=1)

        prb0 = np.squeeze(np.apply_along_axis(two_step_classifier, 1, X_test))

        acc = (np.argmax(prb0, axis=1) == y_test).mean()

        print(k, "Accuracy Final", acc)

        prb[test] = prb0

    acc = (np.argmax(prb, axis=1) == cluster).mean()
    print("Accuracy", acc)

    if save_classification:
        np.save(
            join(
                clf_name,
                f"{participant}_spec_cluster_{num_classes}_{clf_name}_twostage_prb.npy",
            ),
            prb,
        )

    if save_confusion_matrix:

        cm = confusion_matrix(cluster, np.argmax(prb, axis=1), normalize="all")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(
            join(
                clf_name,
                f"{participant}_spec_cluster_{num_classes}_{clf_name}_twostage_cm.png",
            )
        )
        plt.show()

    if save_reconstruction:

        lbl = np.argmax(prb, axis=1)
        centers = np.load(
            join(path_input, f"{participant}_spec_cluster_{num_classes}_centers.npy")
        ).astype(np.float32)
        melSpec_pred = np.stack(tuple(centers[x] for x in lbl), axis=0)

        audiosr = 16000
        centers = get_center(cluster, melSpec, num_classes)
        center_audio = reconstructAudio(lbl, audiosr, centers)
        scipy.io.wavfile.write(
            join(
                clf_name,
                f"{participant}_cluster_{num_classes}_{clf_name}_twostage_center_reconstructed.wav",
            ),
            int(audiosr),
            center_audio,
        )
        pass
