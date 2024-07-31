import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from os.path import join
from os import makedirs
import pandas as pd

master_path = ".."

import sys

sys.path.append(master_path)

import utils

# plt.style.use("seaborn-v0_8")
plt.style.use("ggplot")


def main():
    participants = [f"sub-{i:02d}" for i in range(1, 11)]
    nfolds = 10
    dataset_name = "Word"
    clustering_method = "kmeans"
    vocoder_name = "VocGAN"  # "Griffin_Lim"
    output_path = join(master_path, "results", "comparison_SA", "lossplot")
    results_dir = join(
        master_path, "results", "classification", "40", "Transformer_Encoder"
    )
    nums_classes = (2, 5, 20)

    makedirs(output_path, exist_ok=True)

    EPOCHS = 100

    train_loss = np.full((len(nums_classes), len(participants), nfolds, EPOCHS), np.nan)
    val_loss = np.full((len(nums_classes), len(participants), nfolds, EPOCHS), np.nan)

    for i_participant, participant in enumerate(participants):
        for i_num_classes, num_classes in enumerate(nums_classes):
            file_names = utils.names.Names(
                results_dir,
                dataset_name,
                vocoder_name,
                participant,
                nfolds,
                num_classes,
                clustering_method,
            )
            for fold in range(nfolds):
                file_names.update(fold=fold)
                train_log_file_path = join(file_names.file + "_train_log.csv")
                val_log_file_path = join(file_names.file + "_val_log.csv")

                train_log = pd.read_csv(train_log_file_path)
                val_log = pd.read_csv(val_log_file_path)

                train_total0 = train_log["total"].ravel()
                train_loss0 = train_log["loss"].ravel()
                train_loss[i_num_classes, i_participant, fold, : len(train_total0)] = (
                    train_loss0 / train_total0
                )

                val_total0 = val_log["total"].ravel()
                val_loss0 = val_log["loss"].ravel()
                val_loss[i_num_classes, i_participant, fold, : len(val_total0)] = (
                    val_loss0 / val_total0
                )
    val_loss_argmin = np.nanargmin(val_loss, axis=-1)
    max_epoch = np.max(val_loss_argmin)

    plt.figure(figsize=(10, 4))

    plt.hist(
        val_loss_argmin.ravel(),
        bins=np.arange(max_epoch),
        density=True,
        align="left",
        rwidth=0.8,
    )
    # plt.xlim(0, 4)
    plt.title("Distribution of best epoch in training TE")
    plt.xlabel("Best epoch")
    plt.ylabel("Relative frequency")
    plt.xticks(np.arange(max_epoch))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.savefig(join(output_path, "histogram.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    np.random.seed(5)
    rand_numclass = np.random.randint(len(nums_classes))
    rand_participant = np.random.randint(len(participants))
    rand_fold = np.random.randint(nfolds)
    rand_epoch = np.sum(
        np.logical_not(
            np.isnan(val_loss[rand_numclass, rand_participant, rand_fold, :])
        )
    )

    plt.subplot(2, 1, 1)

    plt.plot(
        np.arange(EPOCHS),
        val_loss[rand_numclass, rand_participant, rand_fold, :],
    )
    plt.plot(
        [val_loss_argmin[rand_numclass, rand_participant, rand_fold]],
        [
            val_loss[
                rand_numclass,
                rand_participant,
                rand_fold,
                val_loss_argmin[rand_numclass, rand_participant, rand_fold],
            ]
        ],
        "ko",
        markersize=5,
    )
    plt.ylabel("Val Loss")
    plt.xticks(np.arange(rand_epoch))

    plt.subplot(2, 1, 2)
    plt.plot(
        np.arange(EPOCHS),
        train_loss[rand_numclass, rand_participant, rand_fold, :],
    )

    plt.ylabel("Train Loss")
    plt.xticks(np.arange(rand_epoch))

    plt.suptitle(f"Loss plot sample")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(join(output_path, "random_loss.png"))
    plt.close()

    print(
        f"par {participants[rand_participant]}, SU {nums_classes[rand_numclass]}, fold {rand_fold}"
    )
    pass


if __name__ == "__main__":
    main()
