from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy
from reconstruction_minimal import createAudio
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram


def hierarchial(n_clusters, melSpec):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = clustering.fit(melSpec)
    return clustering.labels_  # , dists


def kmeans(n_clusters, melSpec):
    clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    clustering.fit_transform(melSpec)
    return clustering.labels_


def cluster(n_clusters, melSpec):
    clustering = hierarchial
    c0 = clustering(2, melSpec)
    silent_ind = 0
    sound_mask = np.where(c0 != silent_ind)
    c1 = clustering(n_clusters - 1, melSpec[sound_mask])
    c0[sound_mask] = c1 + 1
    return c0


def multistage_cluster(n_clusters, melSpec):
    clustering = kmeans
    c0 = clustering(2, melSpec)
    mask = np.where(c0 > 0)
    stages = n_clusters - 2
    for i in range(stages):
        print(i)
        c = clustering(2, melSpec[mask])
        c0[mask] = c + i + 1
        mask = np.where(c0 > i + 1)
    # c = clustering(n_clusters - stages - 1, melSpec[mask])
    # c0[mask] = c + stages + 1
    return c0


def score(lbl, melSpec):
    silhouette_avg = silhouette_score(melSpec, lbl)
    return silhouette_avg


def plot(lbl, melSpec, original_audio, N):
    plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(lbl[:N])
    plt.subplot(3, 1, 2, sharex=ax1)
    plt.imshow(melSpec[:N].T, "gray", aspect="auto")
    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(original_audio[:N])
    plt.tight_layout()


def plot_hist(lbl):
    lbl_hist = np.unique(lbl, return_counts=True)
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


def get_center(lbl, n_clusters):
    center = np.zeros((n_clusters, melSpec.shape[1]))
    for i in range(n_clusters):
        center[i] = np.mean(melSpec[lbl == i], axis=0)
    return center


def reconstructAudio(lbl, audiosr, centers):
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
    path_input = "../Dataset_Sentence"  # "../Dataset_Word"
    participant = "p07_ses1_sentences"  #   "sub-06"
    audiosr = 16000
    save_kmeans = True
    save_reconstructed = False
    plot_figure = False
    print_score = False
    custom_cluster = False

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))

    if custom_cluster:
        lbl = np.load(join(path_input, f"{participant}_spec_cluster_{n_clusters}.npy"))
    else:
        lbl = cluster(n_clusters, melSpec)
    if print_score:
        print(score(lbl, melSpec))

    if save_kmeans:
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}.npy"),
            lbl,
        )
        centers = get_center(lbl, n_clusters)
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_centers.npy"),
            centers,
        )
    if plot_figure:
        original_audio = scipy.io.wavfile.read(
            join(path_input, f"{participant}_orig_synthesized.wav")
        )
        original_audio = scipy.signal.decimate(original_audio[1], 160)
        plot(lbl, melSpec, original_audio, 500)
        # plot_dendrogram(clustering, truncate_mode="level", p=3)
        hist = plot_hist(lbl)
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_hist.npy"), hist
        )

    if save_reconstructed:
        centers = get_center(lbl, n_clusters)
        center_audio = reconstructAudio(lbl, audiosr, centers)
        scipy.io.wavfile.write(
            join(
                path_input,
                f"{participant}_cluster_{n_clusters}_center_reconstructed.wav",
            ),
            int(audiosr),
            center_audio,
        )
    plt.show()
