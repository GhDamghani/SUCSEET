from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy
from reconstruction_minimal import createAudio
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from matplotlib.widgets import Slider


def hierarchial(n_clusters, train, test):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = clustering.fit(train)
    return clustering.predict(test)


def kmeans(n_clusters, train, test):
    clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    clustering.fit(train)
    return clustering.predict(test), clustering.cluster_centers_


def powerfrommelSpec(melSpec):
    return np.square(np.exp(melSpec)).mean()


def get_silence(melSpec):
    power = np.array([powerfrommelSpec(x) for x in melSpec])
    return np.argmin(power)


def get_empty_clusters(lbl, num_clusters):
    return tuple(i for i in range(num_clusters) if i not in lbl)


def get_real_index(lbl):
    values = np.unique(lbl)
    d = dict()
    for i in range(len(values)):
        d[values[i]] = i
    return np.array([d[x] for x in lbl])


def cluster_twostage(n_clusters, melSpec):
    nfolds = 10
    kf = KFold(nfolds, shuffle=False)

    clustering = kmeans

    lbl = np.zeros(melSpec.shape[0], dtype=int)
    melSpec_centers = np.zeros(melSpec.shape)

    for k, (train, test) in enumerate(kf.split(melSpec)):
        trainData = melSpec[train]
        testData = melSpec[test]
        lbl0, centers0 = clustering(2, trainData, testData)
        silence_index = get_silence(centers0)
        speech_mask = np.where(lbl0 != silence_index)
        silence_mask = np.where(lbl0 == silence_index)
        melSpec_centers[test[silence_mask]] = centers0[silence_index]
        lbl1, centers1 = clustering(
            n_clusters - 1, trainData[speech_mask], testData[speech_mask]
        )
        lbl[test[speech_mask]] = 1 + lbl1  # get_real_index(lbl1)
        melSpec_centers[test[speech_mask]] = np.array([centers1[i] for i in lbl1])
        print(
            "K:",
            k,
            "Score:",
            round(score(lbl[test], testData), 3),
            "Empty clusters:",
            get_empty_clusters(lbl[test], n_clusters),
        )
    return lbl, melSpec_centers


def cluster(n_clusters, melSpec):
    nfolds = 10
    kf = KFold(nfolds, shuffle=False)

    clustering = kmeans

    lbl = np.zeros(melSpec.shape[0], dtype=int)

    for k, (train, test) in enumerate(kf.split(melSpec)):
        trainData = melSpec[train]
        testData = melSpec[test]
        lbl0 = clustering(n_clusters, trainData, testData)
        lbl[test] = lbl0
        print(k, round(score(lbl0, testData), 3))

    return lbl


def score(lbl, melSpec):
    silhouette_avg = silhouette_score(melSpec, lbl)
    return silhouette_avg


def plot(lbl, melSpec, original_audio, N):
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(lbl[:N])
    ax[1].imshow(melSpec[:N].T, "gray", aspect="auto")
    ax[2].plot(original_audio[:N])
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(slider_ax, "X Offset", 0, lbl.size - N, valinit=0)

    def update(val):
        # Get the current slider position
        offset = int(slider.val)
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        # Create a new view of the image with the desired offset
        ax[0].plot(lbl[offset : offset + N])
        ax[1].imshow(melSpec[offset : offset + N].T, "gray", aspect="auto")
        ax[2].plot(original_audio[offset : offset + N])
        plt.draw()

    slider.on_changed(update)


def plot_hist(lbl):
    lbl_hist = np.unique(lbl, return_counts=True)
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


if __name__ == "__main__":
    n_clusters = 20
    path_input = "../Dataset_Word"  # "../Dataset_Sentence"
    participant = "sub-06"  #    "p07_ses1_sentences"
    audiosr = 16000
    save_kmeans = True
    save_reconstructed = True
    plot_figure = True
    print_score = True
    custom_cluster = False

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))

    if custom_cluster:
        lbl = np.load(join(path_input, f"{participant}_spec_cluster_{n_clusters}.npy"))
        center_melSpec = np.load(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_centers.npy")
        )
    else:
        lbl, center_melSpec = cluster_twostage(n_clusters, melSpec)

    if save_kmeans:
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}.npy"),
            lbl,
        )
        np.save(
            join(path_input, f"{participant}_spec_cluster_{n_clusters}_centers.npy"),
            center_melSpec,
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
        center_audio = createAudio(center_melSpec, audiosr)
        scipy.io.wavfile.write(
            join(
                path_input,
                f"{participant}_cluster_{n_clusters}_center_reconstructed.wav",
            ),
            int(audiosr),
            center_audio,
        )
    plt.show()
