import scipy
from os.path import join
from os import makedirs
import numpy as np
import classifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
import sys

master_path = ".."
sys.path.append(master_path)

from vocoders.Griffin_Lim import createAudio
from vocoders.VocGAN import StreamingVocGan

import utils


def main():
    num_classes = 2
    participant = "sub-06"  # "p07_ses1_sentences"
    nfolds = 5
    dataset_type = "Word"

    vocoder = "Griffin_Lim"  # "VocGAN"
    clf_name = "LDA"  # "HistGrad"
    method = "onestage"
    clustering_method = "kmeans"

    window_size = 40
    pca_components = 150

    save_reconstruction = False

    path_input = join(master_path, "dataset", dataset_type, vocoder)
    system_type = "SISO" if window_size == 1 else "MISO"

    if system_type == "MISO":
        output_path = join(system_type, f"{window_size}", clf_name)
    else:
        output_path = join(system_type, clf_name)

    makedirs(output_path, exist_ok=True)

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))
    cluster = np.load(
        join(
            master_path,
            "clustering",
            clustering_method,
            f"{participant}_spec_{vocoder}_cluster_{num_classes}_kfold_{nfolds}.npy",
        )
    )
    melSpec_centers = np.load(
        join(
            master_path,
            "clustering",
            clustering_method,
            f"{participant}_spec_{vocoder}_cluster_{num_classes}_kfold_{nfolds}_centers.npy",
        )
    )

    kf = utils.data.WindowedData(
        feat, cluster, window_size=window_size, num_folds=nfolds
    )

    y_pred = dict()

    print(clf_name, pca_components, method, window_size)

    for k, (train, test, whole) in enumerate(kf):
        X_train, y_train = next(train.generate_batch(-1))
        X_test, y_test = next(test.generate_batch(-1))
        X_whole, y_whole = next(whole.generate_batch(-1))

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_whole = X_whole.reshape(X_whole.shape[0], -1)

        clf_fcn = {
            "onestage": classifier.clf_onestage,
            "twostage": classifier.clf_twostage,
        }[method]

        y_pred_prb_train, y_pred_prb_test, y_pred_prb_whole = clf_fcn(
            clf_name,
            X_train,
            X_test,
            y_train,
            y_test,
            X_whole,
            y_whole,
            pca_components,
        )
        y_pred_train = np.argmax(y_pred_prb_train, -1)
        y_pred_test = np.argmax(y_pred_prb_test, -1)
        # y_pred_whole = np.argmax(y_pred_prb_whole, -1)

        acc_test = (y_pred_test == y_test).mean()
        acc_train = (y_pred_train == y_train).mean()
        cluster_hist = np.unique(y_test, return_counts=True)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred_test,
            normalize="all",
        )
        plt.gcf().set_size_inches(8, 8)
        plt.title(
            f"{participant}, {num_classes} classes, {clf_name}, {method} method\n Vocoder: {vocoder} PCA components: {pca_components}\nkfold iteration: {k} Accuracy: {acc_test:.3%} (train: {acc_train:.3%})"
        )
        plt.savefig(
            join(
                output_path,
                f"{participant}_{vocoder}_cluster_{clustering_method}_{num_classes}_{clf_name}_{method}_fold_{k}_confusion.png",
            )
        )
        print(
            k,
            f"train acc {acc_train:.3} test acc {acc_test:.3} max histogram {np.max(cluster_hist[1])/np.sum(cluster_hist[1]):.3}",
        )
        y_pred[f"fold_{k}_y_pred_prb_train"] = y_pred_prb_train
        y_pred[f"fold_{k}_y_pred_prb_test"] = y_pred_prb_test
        y_pred[f"fold_{k}_y_pred_prb_whole"] = y_pred_prb_whole
    np.savez(
        join(
            output_path,
            f"{participant}_{vocoder}_cluster_{clustering_method}_{num_classes}_{clf_name}_{method}_prb",
        ),
        **y_pred,
    )
    if save_reconstruction:
        k = 0
        output_file_name = join(
            output_path,
            f"{participant}_wave_{vocoder}_cluster_{clustering_method}_{num_classes}_{clf_name}_{method}_center_reconstructed.wav",
        )
        center_melSpec = np.stack(
            tuple(
                melSpec_centers[k][x]
                for x in np.argmax(y_pred[f"fold_{k}_y_pred_prb_whole"], -1)
            ),
            axis=0,
        )
        if vocoder == "VocGAN":
            model_path = join(
                master_path, "vocoders", "VocGAN", "vctk_pretrained_model_3180.pt"
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


if __name__ == "__main__":
    main()
