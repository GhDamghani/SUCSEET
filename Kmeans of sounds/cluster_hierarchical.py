from sklearn.cluster import AgglomerativeClustering
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy
from reconstruction_minimal import createAudio
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram


def cluster(n_clusters):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=75)
    clustering = clustering.fit(melSpec)
    return clustering  # , dists


def score(clustering, melSpec):
    silhouette_avg = silhouette_score(melSpec, clustering.labels_)
    return silhouette_avg


def plot(clustering, melSpec, original_audio, N):
    lbl = clustering.labels_
    plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(lbl[:N])
    plt.subplot(3, 1, 2, sharex=ax1)
    plt.imshow(melSpec[:N].T, "gray", aspect="auto")
    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(original_audio[:N])
    plt.tight_layout()


def hist(clustering):
    lbl_hist = np.unique(clustering.labels_, return_counts=True)
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


def reconstructAudio(clustering, audiosr):
    lbl = clustering.labels_
    center = lambda x: clustering.cluster_centers_[x]
    center_melSpec = np.stack(tuple(center(x) for x in lbl), axis=0)
    center_audio = createAudio(center_melSpec, audiosr)
    return center_audio


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure()
    dendrogram(linkage_matrix, **kwargs)


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

    clustering = cluster(n_clusters)
    if print_score:
        print(score(clustering, melSpec))

    if save_kmeans:
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}.npy"),
            clustering.labels_,
        )
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_centers.npy"),
            clustering.cluster_centers_,
        )
    if plot_figure:
        original_audio = scipy.io.wavfile.read(
            join(path_input, f"{participant}_orig_synthesized.wav")
        )
        original_audio = scipy.signal.decimate(original_audio[1], 160)
        plot(clustering, melSpec, original_audio, 500)
        plot_dendrogram(clustering, truncate_mode="level", p=3)
        hist(clustering)

    if save_reconstructed:
        center_audio = reconstructAudio(clustering, audiosr)
        scipy.io.wavfile.write(
            join(path_input, f"{participant}_cluster_center_reconstructed.wav"),
            int(audiosr),
            center_audio,
        )
    plt.show()
