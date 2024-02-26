from sklearn.cluster import AgglomerativeClustering
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy
from reconstruction_minimal import createAudio
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram


def cluster(n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
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


def plot_hist(clustering):
    lbl_hist = np.unique(clustering.labels_, return_counts=True)
    hist = lbl_hist[1]
    np.save("histogram.npy", hist)
    hist_normed = lbl_hist[1] / np.sum(lbl_hist[1])
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
    return lbl_hist[1]


def get_center(clustering):
    lbl = clustering.labels_
    center = np.zeros((clustering.n_clusters, melSpec.shape[1]))
    for i in range(clustering.n_clusters):
        center[i] = np.mean(melSpec[lbl == i], axis=0)
    return center


def reconstructAudio(clustering, audiosr, centers):
    lbl = clustering.labels_
    center_melSpec = np.stack(tuple(centers[x] for x in lbl), axis=0)
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
    path_input = r"../Dataset_Sentence"
    participant = "p07_ses1_sentences"
    audiosr = 16000
    save_kmeans = True
    save_reconstructed = True
    plot_figure = True
    print_score = True

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    if participant == "p07_ses1_sentences":
        noise = np.random.normal(0, 0.0001, (melSpec.shape[0], 1))
        melSpec = melSpec + noise
    feat = np.load(join(path_input, f"{participant}_feat.npy"))

    clustering = cluster(n_clusters)
    if print_score:
        print(score(clustering, melSpec))

    if save_kmeans:
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}.npy"),
            clustering.labels_,
        )
        centers = get_center(clustering)
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_centers.npy"),
            centers,
        )
    if plot_figure:
        original_audio = scipy.io.wavfile.read(
            join(path_input, f"{participant}_orig_synthesized.wav")
        )
        original_audio = scipy.signal.decimate(original_audio[1], 160)
        plot(clustering, melSpec, original_audio, 500)
        # plot_dendrogram(clustering, truncate_mode="level", p=3)
        hist = plot_hist(clustering)
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_hist.npy"), hist
        )

    if save_reconstructed:
        centers = get_center(clustering)
        center_audio = reconstructAudio(clustering, audiosr, centers)
        scipy.io.wavfile.write(
            join(path_input, f"{participant}_cluster_center_reconstructed.wav"),
            int(audiosr),
            center_audio,
        )
    plt.show()
