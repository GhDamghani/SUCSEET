import torch
from model_module import SpeechDecodingModel
from model_loss import criterion
import matplotlib.pyplot as plt
import numpy as np
import data
from trainer import Trainer


def main(config, model_path):

    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)

    _, val_dataset = data.get_train_val_datasets(
        config.feat,
        config.cluster,
        config.timepoints,
        config.num_eeg_channels,
        config.BATCH_SIZE,
        config.EPOCHS,
        config.DEVICE,
        config.VALIDATION_RATIO,
        config.P_SAMPLE,
    )

    model = SpeechDecodingModel(
        config.d_model,
        config.num_heads,
        config.dim_feedforward,
        config.num_layers,
        config.num_classes,
        config.timepoints,
        config.num_eeg_channels,
        config.dropout_prenet,
        config.dropout_encoder,
        config.dropout_clf,
    )
    model.to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))

    criterion = config.criterion()

    trainer = Trainer(
        model,
        criterion,
        None,
        None,
        None,
        val_dataset,
        config.BATCH_SIZE,
        config.EPOCHS,
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
    import config

    main(config, "model.pth")
