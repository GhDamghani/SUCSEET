from sklearn.cluster import KMeans
import numpy as np
from os.path import join
import joblib
import matplotlib.pyplot as plt
import scipy
from reconstruction_minimal import createAudio
from sklearn.metrics import silhouette_score


def cluster(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    dists = kmeans.fit_transform(melSpec)
    return kmeans  # , dists


def score(kmeans, melSpec):
    silhouette_avg = silhouette_score(melSpec, kmeans.labels_)
    return silhouette_avg


def plot(kmeans, melSpec, original_audio, N):
    lbl = kmeans.labels_
    plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(lbl[:N])
    plt.subplot(3, 1, 2, sharex=ax1)
    plt.imshow(melSpec[:N].T, "gray", aspect="auto")
    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(original_audio[:N])
    plt.tight_layout()


def hist(kmeans):
    lbl_hist = np.unique(kmeans.labels_, return_counts=True)
    hist = lbl_hist[1]
    np.save("histogram.npy", hist)
    hist = lbl_hist[1] / np.sum(lbl_hist[1])
    plt.figure()
    plt.stem(
        lbl_hist[0],
        lbl_hist[1] / np.sum(lbl_hist[1]),
    )
    plt.xlabel("Labels")
    plt.ylabel("Freq")
    plt.title("Histogram of Clusters for the whole data")

    plt.tight_layout()
    plt.savefig("histogram.png")


def reconstructAudio(kmeans, audiosr):
    lbl = kmeans.labels_
    center = lambda x: kmeans.cluster_centers_[x]
    center_melSpec = np.stack(tuple(center(x) for x in lbl), axis=0)
    center_audio = createAudio(center_melSpec, audiosr)
    return center_audio


if __name__ == "__main__":
    n_clusters = 20
    path_input = r"../Dataset_Word"
    participant = "sub-06"
    audiosr = 16000
    save_kmeans = False
    save_reconstructed = False
    plot_figure = True
    print_score = True

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))

    kmeans = cluster(n_clusters)
    if print_score:
        print(score(kmeans, melSpec))

    if save_kmeans:
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}.npy"),
            kmeans.labels_,
        )
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_centers.npy"),
            kmeans.cluster_centers_,
        )
    if plot_figure:
        original_audio = scipy.io.wavfile.read(
            join(path_input, f"{participant}_orig_synthesized.wav")
        )
        original_audio = scipy.signal.decimate(original_audio[1], 160)
        plot(kmeans, melSpec, original_audio, 500)
        hist(kmeans)

    if save_reconstructed:
        center_audio = reconstructAudio(kmeans, audiosr)
        scipy.io.wavfile.write(
            join(path_input, f"{participant}_cluster_center_reconstructed.wav"),
            int(audiosr),
            center_audio,
        )
    plt.show()
