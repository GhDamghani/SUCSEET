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


def get_center(lbl, melSpec, n_clusters):
    center = np.zeros((n_clusters, melSpec.shape[1]))
    for i in range(n_clusters):
        mask = lbl == i
        if np.sum(mask) == 0:
            center[i] = 0.0
        else:
            center[i] = np.mean(melSpec[mask], axis=0)
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
    melSpec_center = np.load(
        join(path_input, f"{participant}_spec_cluster_{num_classes}_centers.npy")
    )
    cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))

    clf_name = "QDA"
    CLF = {"QDA": QuadraticDiscriminantAnalysis}
    parameters = {"QDA": {}}

    nfolds = 10
    kf = KFold(nfolds, shuffle=False)

    prb = np.zeros((cluster.shape[0], num_classes), dtype=int)
    melSpec_pred = np.zeros(melSpec_center.shape, dtype=np.float32)

    for k, (train, test) in enumerate(kf.split(feat)):
        X_train, y_train = feat[train], cluster[train]
        X_test, y_test = feat[test], cluster[test]
        centers = get_center(y_test, melSpec_center[test], num_classes)

        clf = CLF[clf_name](**parameters[clf_name])
        clf.fit(X_train, y_train)
        prb0 = clf.predict_proba(X_test)
        pred0 = np.argmax(prb0, axis=1)
        acc = (pred0 == y_test).mean()
        print(k, acc)
        prb[test] = prb0
        melSpec_pred[test] = np.array([centers[i] for i in pred0])

        cm = confusion_matrix(y_test, pred0)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(
            join(
                clf_name,
                f"{participant}_spec_cluster_{num_classes}_{clf_name}_cm_k_{k}.png",
            )
        )

    acc = (np.argmax(prb, axis=1) == cluster).mean()
    print(acc)

    if save_classification:
        np.save(
            join(
                clf_name, f"{participant}_spec_cluster_{num_classes}_{clf_name}_prb.npy"
            ),
            prb,
        )

    if save_reconstruction:

        lbl = np.argmax(prb, axis=1)

        audiosr = 16000
        center_audio = createAudio(melSpec_pred, audiosr)
        scipy.io.wavfile.write(
            join(
                clf_name,
                f"{participant}_cluster_{num_classes}_{clf_name}_center_reconstructed.wav",
            ),
            int(audiosr),
            center_audio,
        )
        pass
