from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy
from os import makedirs

from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch

import sys


master_path = ".."
sys.path.append(master_path)

from vocoders.Griffin_Lim import createAudio
from vocoders.VocGAN import StreamingVocGan


def hierarchial(n_clusters, train, test):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = clustering.fit(train)
    return clustering.predict(test)


def kmeans(n_clusters, train, test, pca_components=None):
    steps = [StandardScaler()]
    if pca_components is not None and isinstance(pca_components, int):
        steps.append(PCA(n_components=pca_components))
    preprocessor = make_pipeline(*steps)
    preprocessed_train = preprocessor.fit_transform(train)

    clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    y = clustering.fit_predict(preprocessed_train)

    centers = get_center(y, train, n_clusters)

    return y, clustering.predict(preprocessor.transform(test)), centers


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


def cluster_twostage(n_clusters, melSpec, train, test, pca_components):
    lbl = np.zeros(melSpec.shape[0], dtype=int)
    lbl0_train, lbl0_test, centers0 = kmeans(
        2, melSpec[train], melSpec[test], pca_components
    )
    silence_index = get_silence(centers0)
    speech_mask_train = np.where(lbl0_train != silence_index)
    speech_mask_test = np.where(lbl0_test != silence_index)
    lbl1_train, lbl1_test, centers1 = kmeans(
        n_clusters - 1,
        melSpec[train[speech_mask_train]],
        melSpec[test[speech_mask_test]],
        pca_components,
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


def cluster(n_clusters, melSpec, train, test, pca_components):

    lbl = np.zeros(melSpec.shape[0], dtype=int)
    lbl_train, lbl_test, centers = kmeans(
        n_clusters, melSpec[train], melSpec[test], pca_components
    )
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
    return lbl_hist[1]


def get_center(lbl, melSpec, n_clusters):
    mean = np.zeros((n_clusters, melSpec.shape[1]))
    for i in range(n_clusters):
        mean[i] = np.mean(melSpec[lbl == i], axis=0)
    dist = np.zeros(
        (
            n_clusters,
            melSpec.shape[0],
        )
    )

    mean_broadcast = np.expand_dims(mean, axis=1)
    melSpec_broadcast = np.expand_dims(melSpec, axis=0)
    dist = np.linalg.norm(melSpec_broadcast - mean_broadcast, axis=2)
    center = np.zeros((n_clusters, melSpec.shape[1]))
    for i in range(n_clusters):
        center[i] = melSpec[np.argmin(dist[i])]
    return center


if __name__ == "__main__":
    n_clusters = 2
    pca_components = 30

    nfolds = 5
    dataset_type = "Word"
    vocoder = "VocGAN"  # "Griffin_Lim"
    path_input = join(master_path, "dataset", dataset_type, vocoder)
    participant = "sub-06"  #    "p07_ses1_sentences"

    save_kmeans = True
    save_reconstructed = True
    plot_figure = True
    custom_cluster = False

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))

    makedirs("kmeans", exist_ok=True)

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

            cluster_fcn = cluster if n_clusters > 2 else cluster_twostage
            lbl[k], melSpec_centers[k] = cluster_fcn(
                n_clusters, melSpec, train, test, pca_components
            )

            print("K", k, "Score", round(score(lbl[k, test], melSpec[test]), 3))

    if save_kmeans:
        np.save(
            join(
                "kmeans",
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_kfold_{nfolds}.npy",
            ),
            lbl,
        )
        np.save(
            join(
                "kmeans",
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_kfold_{nfolds}_centers.npy",
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
            join(
                "kmeans",
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_fold_{k}_hist.npy",
            ),
            hist,
        )
        plt.savefig(
            join(
                "kmeans",
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_fold_{k}_hist.png",
            )
        )

    if save_reconstructed:
        k = 0
        output_file_name = join(
            "kmeans",
            f"{participant}_wave_{vocoder}_cluster_{n_clusters}_fold_{k}_center_reconstructed.wav",
        )
        center_melSpec = np.stack(tuple(melSpec_centers[k][x] for x in lbl[k]), axis=0)
        if vocoder == "VocGAN":
            model_path = join(
                "..", "vocoders", "VocGAN", "vctk_pretrained_model_3180.pt"
            )
            VocGAN = StreamingVocGan(model_path, is_streaming=False)
            center_melSpec = torch.tensor(
                np.transpose(center_melSpec).astype(np.float32)
            )
            waveform_standard, standard_processing_time = (
                VocGAN.mel_spectrogram_to_waveform(mel_spectrogram=center_melSpec)
            )
            StreamingVocGan.save(
                waveform=waveform_standard,
                file_path=output_file_name,
            )
        elif vocoder == "Griffin_Lim":
            audiosr = 16000
            center_audio = createAudio(center_melSpec, audiosr)
            scipy.io.wavfile.write(
                output_file_name,
                int(audiosr),
                center_audio,
            )
    plt.show()
