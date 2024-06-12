from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy
from os import makedirs

from sklearn.metrics import silhouette_score
from matplotlib.widgets import Slider

import torch

import sys


master_path = ".."
sys.path.append(master_path)

from utils.vocoders.Griffin_Lim import createAudio
from utils.vocoders.VocGAN import StreamingVocGan

import utils


def cluster_twostage(n_clusters, X_train, X_test, melSpec_train):
    y_train, y_test, centers = kmeans(2, X_train, X_test, melSpec_train)

    silence_index = get_silence(centers)

    speech_mask_train = np.where(y_train != silence_index)
    speech_mask_test = np.where(y_test != silence_index)

    y1_train, y1_test, centers1 = kmeans(
        n_clusters - 1,
        X_train[speech_mask_train],
        X_test[speech_mask_test],
        melSpec_train[speech_mask_train],
    )

    y_train[speech_mask_train] = 1 + y1_train
    y_test[speech_mask_test] = 1 + y1_test

    centers = np.concatenate((centers[silence_index : silence_index + 1], centers1))

    values = np.unique(y_train)
    l_values = len(values)

    if l_values < n_clusters:
        print("Not enough clusters")

    return y_train, y_test, centers


def cluster(n_clusters, X_train, X_test, melSpec_train):
    y_train, y_test, centers = kmeans(n_clusters, X_train, X_test, melSpec_train)
    return y_train, y_test, centers


def kmeans(n_clusters, X_train, X_test, melSpec_train):

    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    y_train = clustering.fit_predict(X_train)
    y_test = clustering.predict(X_test)
    centers = get_center(n_clusters, y_train, melSpec_train)

    return y_train, y_test, centers


def hierarchial(n_clusters, train, test):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = clustering.fit(train)
    return clustering.predict(test)


def get_center(n_clusters, y, melSpec):
    mean = np.zeros((n_clusters, melSpec.shape[1]))
    for i in range(n_clusters):
        mean[i] = np.mean(melSpec[y == i], axis=0)
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


def score(y, melSpec):
    silhouette_avg = silhouette_score(melSpec, y)
    return silhouette_avg


def powerfrommelSpec(melSpec):
    return np.square(np.exp(melSpec)).mean()


def get_silence(melSpec):
    power = np.array([powerfrommelSpec(x) for x in melSpec])
    return np.argmin(power)


def train(n_clusters, nfolds, participant, melSpec):
    output_dict = {}
    kf = utils.data.WindowedData(
        melSpec,
        melSpec,
        window_size=1,
        num_folds=nfolds,
        output_indices=(-1,),
    )

    scores = np.empty((nfolds,))

    for fold, (train, test) in enumerate(kf):
        X_train, melSpec_train = next(train.generate_batch(-1))
        X_test, _ = next(test.generate_batch(-1))

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        melSpec_train = melSpec_train.reshape(melSpec_train.shape[0], -1)

        cluster_fcn = cluster_twostage if n_clusters > 2 else cluster
        (
            output_dict[f"y_train_{fold:02d}"],
            output_dict[f"y_test_{fold:02d}"],
            output_dict[f"centers_{fold:02d}"],
        ) = cluster_fcn(
            n_clusters,
            X_train,
            X_test,
            melSpec_train,
        )

        scores[fold] = score(output_dict[f"y_test_{fold:02d}"], X_test)

        print(
            f"Participant {participant} Clusters {n_clusters:02d} Fold {fold:02d} Score: {scores[fold]:.3f}"
        )

    return output_dict, scores


def main(miniconfig):
    n_clusters = miniconfig["n_clusters"]

    nfolds = 10
    dataset_name = "Word"
    vocoder_name = "VocGAN"  # "Griffin_Lim"
    path_input = join(master_path, "dataset", dataset_name, vocoder_name)
    participant = miniconfig["participant"]  # "sub-06"   "p07_ses1_sentences"

    save_output = True
    save_reconstructed = True
    save_stats = True

    custom_cluster = False

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    # feat = np.load(join(path_input, f"{participant}_feat.npy"))

    output_path = join(master_path, "results", "clustering", "kmeans")
    makedirs(output_path, exist_ok=True)
    output_file_name = join(
        output_path,
        f"{dataset_name}_{vocoder_name}_{participant}_spec_c{n_clusters:02d}_f{nfolds:02d}",
    )

    if custom_cluster:
        output_dict = np.load(output_file_name + ".npz")
        with np.load(output_file_name + "_stats.npz") as f:
            scores = f["scores"]
    else:
        output_dict, scores = train(n_clusters, nfolds, participant, melSpec)

    centered_melSpec = np.concatenate(
        [
            np.stack(
                tuple(
                    output_dict[f"centers_{fold:02d}"][x]
                    for x in output_dict[f"y_test_{fold:02d}"]
                ),
                axis=0,
            )
            for fold in range(nfolds)
        ],
        axis=0,
    )

    centered_melSpec_random = np.concatenate(
        [
            np.stack(
                tuple(
                    output_dict[f"centers_{fold:02d}"][x]
                    for x in np.random.randint(
                        0, n_clusters, len(output_dict[f"y_test_{fold:02d}"])
                    )
                ),
                axis=0,
            )
            for fold in range(nfolds)
        ],
        axis=0,
    )

    if save_output:
        np.savez_compressed(output_file_name, **output_dict)

    if save_stats:
        correlation = utils.stats.pearson_corr(melSpec, centered_melSpec, axis=1)
        correlation_random = utils.stats.pearson_corr(
            melSpec, centered_melSpec_random, axis=1
        )
        hist_train = np.zeros((nfolds, n_clusters), dtype=int)
        hist_test = np.zeros((nfolds, n_clusters), dtype=int)

        for i_fold in range(nfolds):
            for i_cluster in range(n_clusters):
                hist_train[i_fold, i_cluster] = np.sum(
                    output_dict[f"y_train_{i_fold:02d}"] == i_cluster
                )
                hist_test[i_fold, i_cluster] = np.sum(
                    output_dict[f"y_test_{i_fold:02d}"] == i_cluster
                )
        np.savez_compressed(
            output_file_name + "_stats",
            correlation=correlation,
            correlation_random=correlation_random,
            scores=scores,
            hist_train=hist_train,
            hist_test=hist_test,
        )

    if save_reconstructed:
        reconstruction(vocoder_name, centered_melSpec, output_file_name)
        # reconstruction(
        #     vocoder_name, centered_melSpec_random, output_file_name + "_random"
        # )


def reconstruction(vocoder_name, melSpec, output_file_name):
    if vocoder_name == "VocGAN":
        model_path = join(
            "..", "utils", "vocoders", "VocGAN", "vctk_pretrained_model_3180.pt"
        )
        VocGAN = StreamingVocGan(model_path, is_streaming=False)
        melSpec = torch.tensor(np.transpose(melSpec).astype(np.float32))
        waveform_standard, standard_processing_time = (
            VocGAN.mel_spectrogram_to_waveform(mel_spectrogram=melSpec)
        )
        StreamingVocGan.save(
            waveform=waveform_standard,
            file_path=output_file_name + ".wav",
        )
    elif vocoder_name == "Griffin_Lim":
        audiosr = 16000
        center_audio = createAudio(melSpec, audiosr)
        scipy.io.wavfile.write(
            output_file_name + ".wav",
            int(audiosr),
            center_audio,
        )


if __name__ == "__main__":
    from multiprocessing import Pool
    import time
    from itertools import product

    start_time = time.perf_counter()

    # participants = ["sub-06"]  # [f"sub-{i:02d}" for i in range(1, 11)]
    participants = [f"sub-{i:02d}" for i in range(1, 11) if i != 6]

    n_clusterss = [2, 5, 10, 20]

    miniconfigs = [
        {"participant": participant, "n_clusters": n_clusters}
        for participant, n_clusters in product(participants, n_clusterss)
    ]

    with Pool() as pool:
        pool.map(main, miniconfigs)

    end_time = time.perf_counter()
    print(f"Done! Execution time: {end_time - start_time:.2f} seconds")
