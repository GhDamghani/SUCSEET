import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os import makedirs
from itertools import combinations
from scipy import stats

master_path = ".."

import sys

sys.path.append(master_path)

import utils

# plt.style.use("seaborn-v0_8")
plt.style.use("ggplot")


def compare(
    participants,
    nums_classes,
    nfolds,
    clustering_method,
    results_dir,
    dataset_name,
    vocoder_name,
    output_path,
):
    correlation = np.zeros((len(participants), nfolds, len(nums_classes)))
    for i_participant, participant in enumerate(participants):
        for i_num_classes, num_classes in enumerate(nums_classes):
            with np.load(
                join(
                    results_dir,
                    f"{dataset_name}_{vocoder_name}_{participant}_spec_c{num_classes:02d}_f{nfolds:02d}_stats.npz",
                )
            ) as cluster_stats:

                correlation[i_participant, :, i_num_classes] = cluster_stats[
                    "correlation"
                ]
    correlation = np.reshape(
        correlation, (len(participants) * nfolds, len(nums_classes))
    )

    combs = tuple(combinations(range(len(nums_classes)), 2))
    ttests = np.zeros((len(combs), 2))
    for i_method_permute, x_method_permute in enumerate(combs):
        i = x_method_permute[0]
        j = x_method_permute[1]
        a = correlation[:, i]
        b = correlation[:, j]
        ttests[i_method_permute] = np.array(stats.ttest_ind(a, b))

    fig, ax = plt.subplots(
        layout="constrained",
        figsize=(6, 5),
    )

    ax.set_title("Reconstruction quality by number of speech units")

    ax.violinplot(
        correlation,
        showmeans=True,
    )

    print(np.mean(correlation, axis=0))

    sig_bar_count = 0
    for i_method_permute, x_method_permute in enumerate(combs):
        if ttests[i_method_permute, 1] >= 0.05:
            continue
        elif ttests[i_method_permute, 1] < 0.001:
            # color = "midnightblue"
            text = "***"
        elif ttests[i_method_permute, 1] < 0.01:
            # color = "blue"
            text = "**"
        elif ttests[i_method_permute, 1] < 0.05:
            # color = "mediumslateblue"
            text = "*"

        ax.plot(
            [
                1 + combs[i_method_permute][0],
                1 + combs[i_method_permute][1],
            ],
            [0.95 + sig_bar_count * 0.015] * 2,
            color="k",
            linewidth=1,
        )
        ax.text(
            1 + (combs[i_method_permute][0] + combs[i_method_permute][1]) / 2,
            0.95 + sig_bar_count * 0.015,
            text,
            ha="center",
        )
        sig_bar_count += 1

    ax.set_xticks(np.arange(1, len(nums_classes) + 1))
    ax.set_xticklabels(nums_classes)

    ax.set_xlabel("Number of speech units")
    ax.set_ylabel("r")

    ax.set_ylim(0.7, 1)

    plt.tight_layout()

    file_names = utils.names.Names(
        results_dir,
        dataset_name,
        vocoder_name,
        participant,
        nfolds,
        num_classes,
        clustering_method,
    )
    file_names.update(fold="all")

    fig.savefig(
        join(
            output_path,
            file_names.local_template_no_participant_no_num_classes + ".png",
        ),
        dpi=300,
    )
    plt.close()


def sample_mel_Spectrogram(
    participant,
    nums_classes,
    nfolds,
    clustering_method,
    results_dir,
    dataset_name,
    vocoder_name,
    output_path,
):

    width = 400
    offset = 780

    dataset_dir = join(master_path, "dataset", dataset_name, vocoder_name)

    melSpec = np.load(join(dataset_dir, f"{participant}_spec.npy"))[
        offset : offset + width
    ]

    F = melSpec.shape[1]

    melSpec_list = np.zeros((width, F, len(nums_classes) + 1))

    melSpec_list[..., -1] = melSpec

    for i_num_classes, num_classes in enumerate(nums_classes):
        with np.load(
            join(
                results_dir,
                f"{dataset_name}_{vocoder_name}_{participant}_spec_c{num_classes:02d}_f{nfolds:02d}.npz",
            )
        ) as cluster_data:
            labels = cluster_data["y_test_00"][offset : offset + width]
            centers = cluster_data["centers_00"]
            melSpec_list[..., i_num_classes] = np.stack(
                [centers[label] for label in labels],
            )

    file_names = utils.names.Names(
        results_dir,
        dataset_name,
        vocoder_name,
        participant,
        nfolds,
        2,
        clustering_method,
    )
    file_names.update(fold="all")

    fig, axs = plt.subplots(
        nrows=len(nums_classes) + 1,
        layout="constrained",
        figsize=(7, 7),
    )

    for i_num_classes, num_classes in enumerate(nums_classes):
        axs[i_num_classes].imshow(
            melSpec_list[..., i_num_classes].T,
            cmap="gray",
            aspect="auto",
            origin="lower",
        )
        axs[i_num_classes].set_title(r"$SU={}$".format(num_classes))

    axs[-1].set_title("Original")
    axs[-1].imshow(melSpec_list[..., -1].T, cmap="gray", aspect="auto", origin="lower")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    # fig.suptitle(
    #     f"Reconstructed spectrograms for {participant} from different speech unit numbers"
    # )

    fig.savefig(
        join(
            output_path,
            file_names.local_template_no_participant_no_num_classes
            + "_spec_sample.png",
        ),
        dpi=300,
    )
    plt.close()


def main():

    participants = [f"sub-{i:02d}" for i in range(1, 11)]
    nfolds = 10
    dataset_name = "Word"
    clustering_method = "kmeans"

    vocoder_name = "VocGAN"

    output_path = join(master_path, "results", "comparison_SA", "clustering")
    makedirs(output_path, exist_ok=True)

    results_dir = join(master_path, "results", "clustering", clustering_method)

    nums_classes = (2, 5, 20)

    compare(
        participants,
        nums_classes,
        nfolds,
        clustering_method,
        results_dir,
        dataset_name,
        vocoder_name,
        output_path,
    )

    sample_mel_Spectrogram(
        participants[5],
        nums_classes,
        nfolds,
        clustering_method,
        results_dir,
        dataset_name,
        vocoder_name,
        output_path,
    )


if __name__ == "__main__":
    main()
