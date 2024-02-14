import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
from model.model import SpeechDecodingModel
from utils import prepare_device, read_json
from os.path import join
import joblib
import numpy as np
from data_loader.data_loaders import get_train_val_datasets
from trainer import Trainer
import matplotlib.pyplot as plt

# fix random seeds for reproducibility
SEED = 1379
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    model_path = config.file
    config = read_json(config.config)
    config = argparse.Namespace(**config)

    kmeans_folder = config.kmeans_folder
    path_input = join(kmeans_folder, "features")
    participant = "sub-06"
    cluster = np.load(join(path_input, f"{participant}_kmeans.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config.n_gpu)

    # setup data_loader instances
    _, val_dataset = get_train_val_datasets(
        feat,
        cluster,
        config.timepoints,
        config.num_eeg_channels,
        config.batch_size,
        config.epochs,
        device,
        config.validation_ratio,
        config.p_sample,
    )

    # build model architecture, then print to console
    model = SpeechDecodingModel(
        config.encoder_prenet_out_d,
        config.d_model,
        config.num_heads,
        config.dim_feedforward,
        config.num_classes,
        config.dropout,
        config.enc_hidden_dim,
        config.timepoints,
        config.num_eeg_channels,
    )

    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.load_state_dict(torch.load(model_path))

    criterion = module_loss.lossfn(kmeans_folder, config.num_classes)

    trainer = Trainer(
        model,
        criterion,
        None,
        None,
        None,
        val_dataset,
        config.batch_size,
        config.epochs,
        None,
        None,
        None,
    )

    pred_labels, y_labels = trainer.validate(return_data=True)

    pred_labels_uniques, pred_counts = np.unique(pred_labels, return_counts=True)
    labels, y_hist = np.unique(y_labels, return_counts=True)
    pred_hist = np.zeros(config.num_classes)
    pred_hist[pred_labels_uniques] = pred_counts
    total = np.sum(y_hist)

    pred_hist = pred_hist / total
    y_hist = y_hist / total

    max_plot = max(pred_hist.max(), y_hist.max())

    print(f"Total y Counts: {total}")

    y_labels_seq = y_labels.reshape(-1, config.timepoints)
    pred_labels_seq = pred_labels.reshape(-1, config.timepoints)

    np.random.seed(43450)  # Reproducibility
    no_samples = 8
    samples_ind = np.random.choice(
        np.arange(0, y_labels_seq.shape[0]), no_samples, replace=False
    )
    labels_seq_sample = y_labels_seq[samples_ind]
    pred_labels_seq_sample = pred_labels_seq[samples_ind]
    plt.figure()
    for i in range(no_samples // 4):
        for j in range(4):
            ind = i * 4 + j
            plt.subplot(4, 2, ind + 1)
            plt.plot(
                pred_labels_seq_sample[ind],
                marker="o",
                linestyle="-",
                color="#8B0000",
                label="Predicted",
                alpha=0.5,
            )
            plt.plot(
                labels_seq_sample[ind],
                linestyle="-",
                marker="o",
                color="#AAFF00",
                label="True",
                alpha=0.5,
            )
            plt.ylim(-0.5, config.num_classes - 0.5)
            if ind == 0:
                plt.legend(loc="upper right")

    plt.figure()

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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Testing")
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: config.json)",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="model.pth",
        type=str,
        help="Saved model file path (default: model.pth)",
    )
    config = parser.parse_args()
    main(config)
