from os.path import join
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import scipy
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
    save_reconstruction = False
    num_classes = 20
    feature_folder = "../Dataset_Word"  #  "../Dataset_Sentence"
    path_input = feature_folder  # join(feature_folder, "features")
    participant = "sub-06"  #   "p07_ses1_sentences"
    feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
    cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))

    X_train, X_test, y_train, y_test = train_test_split(
        feat, cluster, test_size=0.2, random_state=0
    )

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
    prb = clf.predict_proba(feat)
    logit = clf.predict_log_proba(feat)
    np.save(
        join(path_input, f"{participant}_spec_cluster_{num_classes}_QDA_logit.npy"),
        logit,
    )

    if save_reconstruction:

        melSpec_pred = clf.predict(feat)
        centers = clf.means_
        melSpec_pred = np.stack(tuple(centers[x] for x in melSpec_pred), axis=0)

        from reconstruction_minimal import createAudio

        audiosr = 16000
        center_audio = createAudio(melSpec_pred, audiosr)
        scipy.io.wavfile.write(
            join(
                "QDA",
                f"{participant}_cluster_{num_classes}_QDA_center_reconstructed.wav",
            ),
            int(audiosr),
            center_audio,
        )
