import torch
from model import SpeechDecodingModel_clf
import loss_metrics
import matplotlib.pyplot as plt
import numpy as np
from trainer import Trainer
from sklearn.metrics import ConfusionMatrixDisplay
from os.path import join
import scipy

import sys

master_path = "../.."
sys.path.append(master_path)

from vocoders.Griffin_Lim import createAudio
from vocoders.VocGAN import StreamingVocGan

import utils


def save_confusion_matrix(y_test, y_pred_test, test_acc, train_acc, config):
    conf_mat_title = f"{config.participant}, {config.num_classes} classes, {config.clf_name}\n Vocoder: {config.vocoder}, PCA components: {config.pca_components}\nkfold iteration: 0 Accuracy: {test_acc:.3%} (train: {train_acc:.3%})"
    output_file_name = f"{config.participant}_{config.vocoder}_cluster_{config.clustering_method}_{config.num_classes}_{config.clf_name}_fold_0_confusion.png"
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_test,
        normalize="all",
    )
    plt.gcf().set_size_inches(8, 8)
    plt.title(conf_mat_title)
    plt.savefig(
        join(
            config.output_path,
            output_file_name,
        )
    )


def save_results_clf(y_pred_train, y_pred_test, y_pred_whole, config):
    y_pred = {}
    y_pred[f"fold_0_y_pred_prb_train"] = y_pred_train
    y_pred[f"fold_0_y_pred_prb_test"] = y_pred_test
    y_pred[f"fold_0_y_pred_prb_whole"] = y_pred_whole
    output_file_name = f"{config.participant}_{config.vocoder}_cluster_{config.clustering_method}_{config.num_classes}_{config.clf_name}_prb"
    np.savez(
        join(
            config.output_path,
            output_file_name,
        ),
        **y_pred,
    )


def save_reconstruction_clf(y_pred_whole, config):
    k = 0
    output_file_name = f"{config.participant}_wave_{config.vocoder}_cluster_{config.clustering_method}_{config.num_classes}_{config.clf_name}_center_reconstructed.wav"
    center_melSpec = np.stack(
        tuple(config.melSpec_centers[k][x] for x in y_pred_whole),
        axis=0,
    )
    if config.vocoder == "VocGAN":
        model_path = join(
            master_path, "vocoders", "VocGAN", "vctk_pretrained_model_3180.pt"
        )
        VocGAN = StreamingVocGan(model_path, is_streaming=False)
        center_melSpec = torch.tensor(np.transpose(center_melSpec).astype(np.float32))
        waveform_standard, standard_processing_time = (
            VocGAN.mel_spectrogram_to_waveform(mel_spectrogram=center_melSpec)
        )
        StreamingVocGan.save(
            waveform=waveform_standard,
            file_path=output_file_name,
        )
    elif config.vocoder == "Griffin_Lim":
        audiosr = 16000
        center_audio = createAudio(center_melSpec, audiosr)
        scipy.io.wavfile.write(
            join(config.output_path, output_file_name),
            int(audiosr),
            center_audio,
        )


def save_histograms(y_pred_test_labels, y_test, test_acc, train_acc, config):

    hist_suptitle = f"{config.participant}, {config.num_classes} classes, {config.clf_name}\n Vocoder: {config.vocoder}, PCA components: {config.pca_components}\nkfold iteration: 0 Accuracy: {test_acc:.3%} (train: {train_acc:.3%})"
    output_file_name = join(
        config.output_path,
        f"{config.participant}_{config.vocoder}_cluster_{config.clustering_method}_{config.num_classes}_{config.clf_name}_fold_0_histogram.png",
    )

    pred_labels_uniques, pred_counts = np.unique(y_pred_test_labels, return_counts=True)
    labels, y_hist = np.unique(y_test, return_counts=True)
    pred_hist = np.zeros(config.num_classes)
    pred_hist[pred_labels_uniques] = pred_counts
    total = np.sum(y_hist)

    pred_hist = pred_hist / total
    y_hist = y_hist / total

    print(labels)
    print(y_hist)
    print(pred_hist)

    max_plot = max(pred_hist.max(), y_hist.max())

    print(f"Total y Counts: {total}")

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.stem(labels, pred_hist, linefmt="-", markerfmt="o", basefmt="k-")
    plt.xticks(range(config.num_classes))
    plt.xlabel("Labels")
    plt.ylabel("Freq")
    plt.ylim(0, max_plot)
    plt.title("Histogram of Predicted Output for test data")

    plt.subplot(2, 1, 2)
    plt.stem(labels, y_hist, linefmt="-", markerfmt="o", basefmt="k-")
    plt.xticks(range(config.num_classes))
    plt.xlabel("Labels")
    plt.ylabel("Freq")
    plt.ylim(0, max_plot)
    plt.title("Histogram of True Output for test data")

    plt.suptitle(hist_suptitle)

    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.show(block=False)


def main(config, model_path):

    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)

    train_dataset, val_dataset, whole_dataset = next(
        iter(
            utils.data.WindowedData(
                config.feat,
                config.cluster,
                window_size=config.window_size,
                num_folds=config.nfolds,
            )
        )
    )

    model = SpeechDecodingModel_clf(
        config.d_model,
        config.num_heads,
        config.dim_feedforward,
        config.num_layers,
        config.num_classes,
        config.window_size,
        config.num_eeg_channels,
        config.dropout,
    )
    model.to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))

    loss = loss_metrics.get_loss(train_dataset, config.num_classes)

    trainer = Trainer(
        model,
        loss,
        config.metrics,
        val_dataset,
        config.model_task,
        batch_size=config.BATCH_SIZE * 4,
        device=config.DEVICE,
    )
    y_pred_test, y_test, test_metrics = trainer.validate(return_data=True)

    trainer.val_dataset = train_dataset
    y_pred_train, y_train, train_metrics = trainer.validate(return_data=True)

    trainer.val_dataset = whole_dataset
    y_pred_whole, y_whole, whole_metrics = trainer.validate(return_data=True)

    if config.model_task == "classification":
        y_pred_test_labels = np.argmax(y_pred_test, -1)
        y_pred_whole_labels = np.argmax(y_pred_whole, -1)
        test_acc = test_metrics["corrects"] / test_metrics["total"]
        train_acc = train_metrics["corrects"] / train_metrics["total"]

        print(f"Accuracy: {test_acc:02.2%}")
        save_confusion_matrix(y_test, y_pred_test_labels, test_acc, train_acc, config)
        save_histograms(y_pred_test_labels, y_test, test_acc, train_acc, config)
        save_results_clf(y_pred_train, y_pred_test, y_pred_whole, config)
        save_reconstruction_clf(y_pred_whole_labels, config)
        plt.show()


if __name__ == "__main__":
    import config

    config.config_proc(config.Config)

    config = config.Config

    main(config, "model.pth")
