from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy
from reconstruction_minimal import createAudio
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def tsne(n_clusters, train, test):
    tsne = TSNE(n_components=3)
    y_pred = tsne.fit_transform(train)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2])
    plt.show()
    pass


def hierarchial(n_clusters, train, test):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = clustering.fit(train)
    return clustering.predict(test)


def kmeans(n_clusters, numComps, train, test):
    pca = PCA()
    pca.fit(train)

    trainData = np.dot(train, pca.components_[:numComps, :].T)
    testData = np.dot(test, pca.components_[:numComps, :].T)
    clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    y = clustering.fit_predict(trainData)
    centers = np.stack([np.mean(train[y == i], axis=0) for i in range(n_clusters)])
    return y, clustering.predict(testData), centers


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


def cluster_twostage(n_clusters, melSpec, train, test):
    numComps = 15
    lbl = np.zeros(melSpec.shape[0], dtype=int)
    lbl0_train, lbl0_test, centers0 = kmeans(2, numComps, melSpec[train], melSpec[test])
    silence_index = get_silence(centers0)
    speech_mask_train = np.where(lbl0_train != silence_index)
    speech_mask_test = np.where(lbl0_test != silence_index)
    lbl1_train, lbl1_test, centers1 = kmeans(
        n_clusters - 1,
        numComps,
        melSpec[train[speech_mask_train]],
        melSpec[test[speech_mask_test]],
    )
    lbl0_test[speech_mask_test] = 1 + lbl1_test
    lbl0_train[speech_mask_train] = 1 + lbl1_train
    lbl[train] = lbl0_train
    lbl[test] = lbl0_test

    centers = np.concatenate((centers0[silence_index : silence_index + 1], centers1))

    values = np.unique(lbl)
    l_values = len(values)
    if l_values < n_clusters:
        print(k, "Not enough clusters")
        print("Before", values)
        d = dict()
        for i in range(len(values)):
            d[values[i]] = i
        lbl = np.array([d[x] for x in lbl])
        print("After", np.unique(lbl))
        centers_corrected = np.zeros((n_clusters, melSpec.shape[1]))
        centers_corrected[list(range(l_values))] = centers[values]
        print(np.all(centers_corrected[l_values:] == 0.0))
    else:
        centers_corrected = centers

    return lbl, centers_corrected


def cluster(n_clusters, melSpec, train, test):

    lbl = np.zeros(melSpec.shape[0], dtype=int)
    lbl_train, lbl_test, centers = kmeans(n_clusters, melSpec[train], melSpec[test])
    lbl[train] = lbl_train
    lbl[test] = lbl_test

    return lbl, centers


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
    n_clusters = 5
    nfolds = 10
    path_input = "../Dataset_Word"  # "../Dataset_Sentence"
    participant = "sub-06"  #    "p07_ses1_sentences"
    audiosr = 16000
    save_kmeans = True
    save_reconstructed = True
    plot_figure = True
    custom_cluster = False

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))

    if custom_cluster:
        lbl = np.load(
            join(
                "kmeans", f"{participant}_spec_cluster_{n_clusters}_kfold_{nfolds}.npy"
            )
        )
        melSpec_centers = np.load(
            join(
                "kmeans",
                f"{participant}_spec_cluster_{n_clusters}_kfold_{nfolds}_centers.npy",
            )
        )
    else:

        kf = KFold(nfolds, shuffle=False)

        lbl = np.zeros((nfolds, melSpec.shape[0]), dtype=int)
        melSpec_centers = np.zeros([nfolds, n_clusters, melSpec.shape[1]])
        for k, (train, test) in enumerate(kf.split(melSpec)):
            lbl[k], melSpec_centers[k] = cluster_twostage(
                n_clusters, melSpec, train, test
            )
            print("K", k, "Score", round(score(lbl[k, test], melSpec[test]), 3))

    if save_kmeans:
        np.save(
            join(
                "kmeans", f"{participant}_spec_cluster_{n_clusters}_kfold_{nfolds}.npy"
            ),
            lbl,
        )
        np.save(
            join(
                "kmeans",
                f"{participant}_spec_cluster_{n_clusters}_kfold_{nfolds}_centers.npy",
            ),
            melSpec_centers,
        )
    if plot_figure:
        k = 0
        original_audio = scipy.io.wavfile.read(
            join(path_input, f"{participant}_orig_synthesized.wav")
        )
        original_audio = scipy.signal.decimate(original_audio[1], 160)
        plot(lbl[k], melSpec, original_audio, 500)
        # plot_dendrogram(clustering, truncate_mode="level", p=3)
        hist = plot_hist(lbl[k])
        np.save(
            join("kmeans", f"{participant}_spec_cluster_{n_clusters}_k_{k}_hist.npy"),
            hist,
        )

    if save_reconstructed:
        k = 0
        center_melSpec = np.stack(tuple(melSpec_centers[k][x] for x in lbl[k]), axis=0)
        center_audio = createAudio(center_melSpec, audiosr)
        scipy.io.wavfile.write(
            join(
                "kmeans",
                f"{participant}_cluster_{n_clusters}_k_{k}_center_reconstructed.wav",
            ),
            int(audiosr),
            center_audio,
        )
    plt.show()
