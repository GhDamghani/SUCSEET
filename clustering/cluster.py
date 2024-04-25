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

import utils


def hierarchial(n_clusters, train, test):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = clustering.fit(train)
    return clustering.predict(test)


def kmeans(n_clusters, X_train, X_whole, melSpec_train):

    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    y = clustering.fit_predict(X_train)

    centers = get_center(y, melSpec_train, n_clusters)

    return y, clustering.predict(X_whole), centers


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


def cluster_twostage(n_clusters, X_train, X_whole, melSpec_train):
    lbl0_train, lbl0_whole, centers0 = kmeans(2, X_train, X_whole, melSpec_train)
    silence_index = get_silence(centers0)
    speech_mask_train = np.where(lbl0_train != silence_index)
    speech_mask_whole = np.where(lbl0_whole != silence_index)
    lbl1_train, lbl1_whole, centers1 = kmeans(
        n_clusters - 1,
        X_train[speech_mask_train],
        X_whole[speech_mask_whole],
        melSpec_train[speech_mask_train],
    )
    lbl0_train[speech_mask_train] = 1 + lbl1_train
    lbl0_whole[speech_mask_whole] = 1 + lbl1_whole

    centers = np.concatenate((centers0[silence_index : silence_index + 1], centers1))

    lbl = lbl0_whole
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
        centers_corrected = np.zeros((n_clusters, X_whole.shape[1]))
        centers_corrected[list(range(l_values))] = centers[values]
        print(np.all(centers_corrected[l_values:] == 0.0))
    else:
        centers_corrected = centers

    return lbl, centers_corrected


def cluster(n_clusters, X_train, X_whole, melSpec_train):
    lbl_train, lbl, centers = kmeans(n_clusters, X_train, X_whole, melSpec_train)
    return lbl, centers


def score(lbl, melSpec):
    silhouette_avg = silhouette_score(melSpec, lbl)
    return silhouette_avg


def plot(lbl, melSpec, original_audio, N, ratio):
    fig, ax = plt.subplots(3, 1, figsize=(5, 5))
    offset = 3130
    plot_update(lbl, melSpec, original_audio, N, ax, offset, ratio)
    slider_ax = fig.add_axes([0.2, 0, 0.6, 0.03])
    slider = Slider(slider_ax, "X Offset", 0, lbl.size - N, valinit=0)

    def update(val):
        # Get the current slider position
        offset = int(slider.val)
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        # Create a new view of the image with the desired offset
        plot_update(lbl, melSpec, original_audio, N, ax, offset, ratio)
        plt.draw()

    # slider.on_changed(update)
    plt.tight_layout()
    plt.savefig("demo.png", transparent=True)


color_pallet = [
    "#808080",
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]
color_pallet = [
    (int(x[1:3], 16) / 255, int(x[3:5], 16) / 255, int(x[5:7], 16) / 255)
    for x in color_pallet
]


def plot_update(lbl, melSpec, original_audio, N, ax, offset, ratio):
    class_image = np.zeros((1, N, 3))
    for i in range(N):
        class_image[0, i] = color_pallet[lbl[offset + i]]
    ax[0].plot(original_audio[offset * ratio : (offset + N) * ratio])
    ax[0].set_xlim(0, N * ratio)
    ax[0].set_ylim(-1.0, 1.0)
    ax[0].axis("off")
    ax[1].imshow(melSpec[offset : offset + N].T, "gray", aspect="auto")
    ax[1].axis("off")
    ax[2].imshow(class_image, aspect="auto")
    ax[2].axis("off")


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
    n_clusters = 10
    pca_components = None

    nfolds = 10
    dataset_type = "Word"
    vocoder = "VocGAN"  # "Griffin_Lim"
    path_input = join(master_path, "dataset", dataset_type, vocoder)
    participant = "sub-06"  #    "p07_ses1_sentences"

    DCT_coeffs = 40
    preprocessing_list = ["normalize"]
    if DCT_coeffs is not None:
        preprocessing_list.append("DCT")

    save_kmeans = True
    save_reconstructed = True
    plot_figure = False
    custom_cluster = False

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))

    output_path = join(master_path, "results", "clustering", "kmeans")
    makedirs(output_path, exist_ok=True)

    if custom_cluster:
        lbl = np.load(
            join(
                output_path,
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_kfold_{nfolds}.npy",
            )
        )
        melSpec_centers = np.load(
            join(
                output_path,
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_kfold_{nfolds}_centers.npy",
            )
        )
    else:

        kf = utils.data.WindowedData(
            melSpec,
            melSpec,
            window_size=1,
            num_folds=nfolds,
            output_size=-1,
            DCT_coeffs=DCT_coeffs,
            preprocessing=preprocessing_list,
        )

        lbl = np.zeros((nfolds, melSpec.shape[0]), dtype=int)
        melSpec_centers = np.zeros([nfolds, n_clusters, melSpec.shape[1]])
        for k, (train, test, whole) in enumerate(kf):
            X_train, melSpec_train = next(train.generate_batch(-1))
            X_whole, _ = next(whole.generate_batch(-1))

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_whole = X_whole.reshape(X_whole.shape[0], -1)

            melSpec_train = melSpec_train.reshape(melSpec_train.shape[0], -1)

            cluster_fcn = cluster_twostage if n_clusters > 2 else cluster
            lbl[k], melSpec_centers[k] = cluster_fcn(
                n_clusters,
                X_train,
                X_whole,
                melSpec_train,
            )
            correlation = None

            # center_melSpec = np.stack(
            #     tuple(melSpec_centers[k][x] for x in lbl[k]), axis=0
            # )
            # corrs = []
            # for i in range(melSpec.shape[0]):
            #     corrs.append(scipy.stats.pearsonr(melSpec[i], center_melSpec[i])[0])
            # correlation = np.mean(corrs)

            corr_s = f"Mean Correlation: {correlation:.3f}" if correlation else ""

            print(f"K {k} Score: {score(lbl[k], X_whole):.3f}" + corr_s)

    if save_kmeans:
        np.save(
            join(
                output_path,
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_kfold_{nfolds}.npy",
            ),
            lbl,
        )
        np.save(
            join(
                output_path,
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_kfold_{nfolds}_centers.npy",
            ),
            melSpec_centers,
        )
    if plot_figure:
        k = 0
        original_audio = scipy.io.wavfile.read(
            join(path_input, f"{participant}_orig_synthesized.wav")
        )
        original_audio = original_audio[1] / 2**15
        ratio = original_audio.shape[0] // X_whole.shape[0]

        plot(lbl[k], X_whole, original_audio, 400, ratio)
        hist = plot_hist(lbl[k])
        np.save(
            join(
                output_path,
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_fold_{k}_hist.npy",
            ),
            hist,
        )
        plt.savefig(
            join(
                output_path,
                f"{participant}_spec_{vocoder}_cluster_{n_clusters}_fold_{k}_hist.png",
            )
        )

    if save_reconstructed:
        k = 0
        output_file_name = join(
            output_path,
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
