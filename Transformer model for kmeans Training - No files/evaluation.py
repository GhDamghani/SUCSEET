import numpy as np
from os.path import join, exists
from os import listdir
import torch
from model import SpeechDecodingModel
from train_step import evaluate_model
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Namespace:
    pass


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    window_width = 96

    batch_size = 16
    num_classes = 20

    d_model = 128
    num_heads = 4
    dim_feedforward = 256
    num_layers = 3
    dropout = 0.1

    model = SpeechDecodingModel(
        d_model, num_classes, num_heads, dropout, num_layers, dim_feedforward
    ).to(device)

    print(model)
    model.load_state_dict(torch.load("model.pth"))

    model_namespace = Namespace()
    model_namespace.model = model
    model_namespace.device = device
    model_namespace.num_classes = num_classes
    model_namespace.lossfn = torch.nn.CrossEntropyLoss(reduction="sum")
    data_namespace = Namespace()
    data_namespace.test_data_path = "test_data"
    data_namespace.no_test_batches = len(listdir(data_namespace.test_data_path))
    preprocess_namespace = Namespace()
    create_test_files = False
    logger = print

    test_loss, test_acc, pred_labels, y_labels = evaluate_model(
        model_namespace,
        data_namespace,
        preprocess_namespace,
        create_test_files,
        logger,
        True,
    )

    pred_labels_uniques, pred_counts = np.unique(pred_labels, return_counts=True)
    labels, y_hist = np.unique(y_labels, return_counts=True)
    pred_hist = np.zeros(num_classes)
    pred_hist[pred_labels_uniques] = pred_counts

    print(f"Total y Counts: {np.sum(y_hist)}")

    y_labels_seq = y_labels.reshape(-1, window_width)
    pred_labels_seq = pred_labels.reshape(-1, window_width)

    np.random.seed(234)  # Reproducibility
    no_samples = 3
    samples_ind = np.random.choice(
        np.arange(0, y_labels_seq.shape[0]), no_samples, replace=False
    )
    labels_seq_sample = y_labels_seq[samples_ind]
    # pred_labels_seq_sample = pred_labels_seq[samples_ind]
    no_disturbance = window_width // 10
    pred_labels_seq_sample = labels_seq_sample.copy()
    np.random.seed(2834)  # Reproducibility
    disturbance = np.random.choice(
        np.arange(0, window_width), (no_samples, no_disturbance), replace=True
    )
    for i in range(no_samples):
        for j in range(no_disturbance):
            pred_labels_seq_sample[i, disturbance[i, j]] = np.random.randint(
                0, num_classes
            )

    plt.figure()
    for i in range(no_samples):
        plt.subplot(no_samples, 1, i + 1)
        plt.plot(
            pred_labels_seq_sample[i],
            marker="o",
            linestyle="-",
            color="#8B0000",
            label="Predicted",
            alpha=0.5,
        )
        plt.plot(
            labels_seq_sample[i],
            linestyle="-",
            marker="o",
            color="#AAFF00",
            label="True",
            alpha=0.5,
        )
        plt.legend(loc="upper right")

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.stem(labels, pred_hist, linefmt="-", markerfmt="o", basefmt="k-")
    plt.xticks(range(num_classes))
    plt.xlabel("Labels")
    plt.ylabel("Freq")
    plt.title("Histogram of Predicted Output for test data")

    plt.subplot(2, 1, 2)
    plt.stem(labels, y_hist, linefmt="-", markerfmt="o", basefmt="k-")
    plt.xticks(range(num_classes))
    plt.xlabel("Labels")
    plt.ylabel("Freq")
    plt.title("Histogram of True Output for test data")

    plt.tight_layout()
    plt.show()
