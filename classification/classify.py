import scipy
from os.path import join
from os import makedirs
import numpy as np
import classifier

from sklearn.model_selection import KFold

import torch
import sys

master_path = ".."
sys.path.append(master_path)

from utils.vocoders.Griffin_Lim import createAudio
from utils.vocoders.VocGAN import StreamingVocGan

import utils


def save_post_results(
    file_names, nfolds, num_classes, participant, clf_name, vocoder_name, topk
):
    fold = "all"
    file_names.update(fold)
    results = np.load(file_names.results + ".npy", allow_pickle=True).item()
    stats = np.load(file_names.stats + ".npy", allow_pickle=True).item()
    for fold in range(nfolds):
        file_names.update(fold)
        y_test = results["y_test"][fold]
        y_pred_test = results["y_pred_test"][fold]
        train_acc = stats["train_corrects"][fold] / stats["train_total"][fold]
        test_acc = stats["test_corrects"][fold] / stats["test_total"][fold]
        test_speech_acc = (
            stats["test_corrects_speech"][fold] / stats["test_total_speech"][fold]
        )
        if num_classes > 2:
            test_topk_acc = (
                stats["test_topk_corrects"][fold] / stats["test_total"][fold]
            )
            test_topk_speech_acc = (
                stats["test_topk_corrects_speech"][fold]
                / stats["test_total_speech"][fold]
            )
        else:
            test_topk_acc = None
            test_topk_speech_acc = None
        y_pred_test_labels = np.argmax(y_pred_test, axis=1)
        # centred_melSpec = results["centred_melSpec_test"][fold]
        utils.stats.save_confusion_matrix(
            y_test,
            y_pred_test_labels,
            test_acc,
            test_speech_acc,
            test_topk_acc,
            test_topk_speech_acc,
            participant,
            num_classes,
            clf_name,
            vocoder_name,
            fold,
            topk,
            file_names.confusion,
        )
        utils.stats.save_histograms(
            y_test,
            y_pred_test_labels,
            train_acc,
            test_acc,
            test_speech_acc,
            test_topk_acc,
            test_topk_speech_acc,
            participant,
            num_classes,
            clf_name,
            vocoder_name,
            fold,
            topk,
            file_names.histogram,
        )


def save_stats_summary(file_name, num_classes):
    fold = "all"
    file_name.update(fold)
    with np.load(file_name.stats + ".npz") as f:
        stats = {key: f[key] for key in f.keys()}

    train_accuracy = stats["train_corrects"].sum() / stats["train_total"].sum()
    test_accuracy = stats["test_corrects"].sum() / stats["test_total"].sum()
    train_speech_accuracy = (
        stats["train_corrects_speech"].sum() / stats["train_total_speech"].sum()
    )
    test_speech_accuracy = (
        stats["test_corrects_speech"].sum() / stats["test_total_speech"].sum()
    )
    if num_classes > 2:
        train_topk_accuracy = (
            stats["train_topk_corrects"].sum() / stats["train_total"].sum()
        )
        test_topk_accuracy = (
            stats["test_topk_corrects"].sum() / stats["test_total"].sum()
        )
    else:
        train_topk_accuracy = None
        test_topk_accuracy = None
    with open(file_name.stats + ".txt", "w") as f:
        f.write(f"train_accuracy: {train_accuracy}\n")
        f.write(f"test_accuracy: {test_accuracy}\n")
        f.write(f"train_speech_accuracy: {train_speech_accuracy}\n")
        f.write(f"test_speech_accuracy: {test_speech_accuracy}\n")
        if num_classes > 2:
            f.write(f"train_topk_accuracy: {train_topk_accuracy}\n")
            f.write(f"test_topk_accuracy: {test_topk_accuracy}\n")


def train_and_save_clf(
    file_names,
    clf_name,
    clf_dict,
    kf,
    method,
    window_size,
    pca_components,
    melSpec_centers,
):
    clf_data = {
        "y_train": [],
        "y_test": [],
        "y_pred_train": [],
        "y_pred_test": [],
        "centred_melSpec_train": [],
        "centred_melSpec_test": [],
    }
    print(
        f"Classifier: {clf_name} PCA: {pca_components}, Method: {method}, Window Size: {window_size}"
    )
    for k, (train, test) in enumerate(kf):
        X_train, y_train = next(train.generate_batch(-1))
        X_test, y_test = next(test.generate_batch(-1))

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        y_train = y_train.ravel()
        y_test = y_test.ravel()

        clf_fcn = clf_dict[method]

        y_pred_train, y_pred_test = clf_fcn(
            clf_name,
            X_train,
            X_test,
            y_train,
            y_test,
            pca_components,
        )
        y_pred_label_train = np.argmax(y_pred_train, -1)
        y_pred_label_test = np.argmax(y_pred_test, -1)

        acc_test = (y_pred_label_test == y_test).mean()
        acc_train = (y_pred_label_train == y_train).mean()
        cluster_hist = np.unique(y_test, return_counts=True)
        print(
            k,
            f"train acc {acc_train:.3} test acc {acc_test:.3} max histogram {np.max(cluster_hist[1])/np.sum(cluster_hist[1]):.3}",
        )

        center_melSpec_train = np.stack(
            tuple(melSpec_centers[k][x] for x in y_pred_label_train),
            axis=0,
        )
        center_melSpec_test = np.stack(
            tuple(melSpec_centers[k][x] for x in y_pred_label_test),
            axis=0,
        )

        clf_data["y_train"].append(y_train)
        clf_data["y_test"].append(y_test)
        clf_data["y_pred_train"].append(y_pred_train)
        clf_data["y_pred_test"].append(y_pred_test)
        clf_data["centred_melSpec_train"].append(center_melSpec_train)
        clf_data["centred_melSpec_test"].append(center_melSpec_test)
    np.save(
        file_names.results,
        clf_data,
        allow_pickle=True,
    )


def main(miniconfig):
    num_classes = miniconfig["num_classes"]  # 20
    participant = miniconfig["participant"]  # "sub-06" "p07_ses1_sentences"
    nfolds = 10
    dataset_name = "Word"

    vocoder_name = "VocGAN"  # "Griffin_Lim"
    clf_name = "LDA"  # "AdaBoost" "HistGrad"
    method = "onestage"
    clf_dict = {
        "onestage": classifier.clf_onestage,
        "twostage": classifier.clf_twostage,
    }
    clustering_method = "kmeans"

    window_size = 1
    topk = 3
    pca_components = None

    save_reconstruction = False

    dataset_path = join(master_path, "dataset", dataset_name, vocoder_name)
    cluster_path = join(
        master_path,
        "results",
        "clustering",
        clustering_method,
    )

    output_path = join(
        master_path,
        "results",
        "classification",
        f"{window_size}",
        clf_name,
    )

    file_names = utils.names.Names(
        output_path,
        dataset_name,
        vocoder_name,
        participant,
        nfolds,
        num_classes,
        clustering_method,
    )
    file_names.update(fold="all")

    makedirs(file_names.output_path, exist_ok=True)

    melSpec = np.load(join(dataset_path, f"{participant}_spec.npy"))
    feat = np.load(join(dataset_path, f"{participant}_feat.npy"))
    cluster = np.empty(
        (
            nfolds,
            feat.shape[0],
        )
    )

    with np.load(
        join(
            cluster_path,
            f"{dataset_name}_{vocoder_name}_{participant}_spec_c{num_classes:02d}_f{nfolds:02d}.npz",
        )
    ) as custering_data:
        melSpec_centers = np.stack(
            [custering_data[f"centers_{i:02d}"] for i in range(nfolds)]
        )
        for fold, (train_index, test_index) in enumerate(
            KFold(n_splits=nfolds, shuffle=False).split(feat)
        ):
            cluster[fold, train_index] = custering_data["y_train_{0:02d}".format(fold)]
            cluster[fold, test_index] = custering_data["y_test_{0:02d}".format(fold)]

    kf = utils.data.WindowedData(
        feat,
        cluster,
        window_size=window_size,
        num_folds=nfolds,
    )

    train_and_save_clf(
        file_names,
        clf_name,
        clf_dict,
        kf,
        method,
        window_size,
        pca_components,
        melSpec_centers,
    )

    clf_data = np.load(file_names.results + ".npy", allow_pickle=True).item()

    melSpecs_train = []
    melSpecs_test = []
    for fold, (train_index, test_index) in enumerate(
        KFold(n_splits=nfolds, shuffle=False).split(feat)
    ):
        melSpecs_train.append(melSpec[train_index])
        melSpecs_test.append(melSpec[test_index])
    utils.stats.save_stats(
        clf_data,
        melSpecs_train,
        melSpecs_test,
        window_size,
        nfolds,
        num_classes,
        topk,
        file_names.stats,
    )
    save_post_results(
        file_names, nfolds, num_classes, participant, clf_name, vocoder_name, topk
    )
    utils.stats.save_stats_summary(file_names, num_classes)
    if save_reconstruction:
        for fold, (train_index, test_index) in enumerate(
            KFold(n_splits=nfolds, shuffle=False).split(feat)
        ):
            centered_melSpec = np.empty((feat.shape[0], melSpec_centers.shape[1]))
            centered_melSpec[train_index] = clf_data["centred_melSpec_train"][fold]
            centered_melSpec[test_index] = clf_data["centred_melSpec_test"][fold]

            output_file_name = join(
                output_path,
                f"{dataset_name}_{vocoder_name}_{participant}_cluster_{clustering_method}_c{num_classes:02d}_f{nfolds:02d}_{fold:02d}.wav",
            )
            if vocoder_name == "VocGAN":
                model_path = join(
                    master_path, "vocoders", "VocGAN", "vctk_pretrained_model_3180.pt"
                )
                VocGAN = StreamingVocGan(model_path, is_streaming=False)
                centered_melSpec = torch.tensor(
                    np.transpose(centered_melSpec).astype(np.float32)
                )
                waveform_standard, standard_processing_time = (
                    VocGAN.mel_spectrogram_to_waveform(mel_spectrogram=centered_melSpec)
                )
                StreamingVocGan.save(
                    waveform=waveform_standard,
                    file_path=output_file_name,
                )
            elif vocoder_name == "Griffin_Lim":
                audiosr = 16000
                center_audio = createAudio(centered_melSpec, audiosr)
                scipy.io.wavfile.write(
                    output_file_name,
                    int(audiosr),
                    center_audio,
                )
            break


if __name__ == "__main__":
    from multiprocessing import Pool
    import time
    from itertools import product

    start_time = time.perf_counter()

    participants = [f"sub-{i:02d}" for i in range(1, 11) if i != 6]

    nums_classes = [2, 5, 10, 20]

    miniconfigs = [
        {"participant": participant, "num_classes": num_classes}
        for participant, num_classes in product(participants, nums_classes)
    ]

    # main(miniconfigs[0])
    for miniconfig in miniconfigs:
        main(miniconfig)

    # with Pool() as pool:
    #     pool.map(main, miniconfigs)
    end_time = time.perf_counter()
    print(f"Done! Execution time: {end_time - start_time:.2f} seconds")
