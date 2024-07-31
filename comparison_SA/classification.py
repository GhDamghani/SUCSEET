import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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
    methods,
    participants,
    num_classes,
    nfolds,
    clustering_method,
    results_dir,
    dataset_name,
    vocoder_name,
    topk,
    output_path,
):
    num_metrics = 2
    if num_classes > 2:
        num_metrics += 3
    for method in methods:
        method["stats"] = {}
        method["metrics"] = {}
        for participant in participants:
            file_names = utils.names.Names(
                join(results_dir, method["dir"]),
                dataset_name,
                vocoder_name,
                participant,
                nfolds,
                num_classes,
                clustering_method,
            )
            file_names.update(fold="all")
            method["stats"][participant] = np.load(
                file_names.stats + ".npy", allow_pickle=True
            ).item()

            test_accuracy = np.array(
                method["stats"][participant]["test_corrects"]
            ) / np.array(method["stats"][participant]["test_total"])

            test_melSpec_corr = method["stats"][participant]["test_melSpec_corr"]

            method["metrics"][participant] = [test_accuracy, test_melSpec_corr]
            if num_classes > 2:
                test_topk_accuracy = np.array(
                    np.array(method["stats"][participant]["test_topk_corrects"])
                    / np.array(method["stats"][participant]["test_total"])
                )
                test_speech_accuracy = np.array(
                    np.array(method["stats"][participant]["test_corrects_speech"])
                    / np.array(method["stats"][participant]["test_total_speech"])
                )
                test_topk_speech_accuracy = np.array(
                    np.array(method["stats"][participant]["test_topk_corrects_speech"])
                    / np.array(method["stats"][participant]["test_total_speech"])
                )
                method["metrics"][participant].insert(1, test_topk_accuracy)
                method["metrics"][participant].insert(2, test_speech_accuracy)
                method["metrics"][participant].insert(3, test_topk_speech_accuracy)

    combs = tuple(combinations(range(len(methods)), 2))
    ttests = np.zeros((num_metrics, len(combs), 2))
    for i_metric in range(num_metrics):
        for i_method_permute, x_method_permute in enumerate(combs):
            i = x_method_permute[0]
            j = x_method_permute[1]
            a = np.concatenate(
                [
                    methods[i]["metrics"][participant][i_metric]
                    for participant in participants
                ]
            )
            b = np.concatenate(
                [
                    methods[j]["metrics"][participant][i_metric]
                    for participant in participants
                ]
            )
            ttests[i_metric, i_method_permute] = np.array(stats.ttest_ind(a, b))

    methods_size = len(methods)
    width = 0.5 / methods_size

    xticks = ["Accuracy", "MelSpec Corr"]

    baseline = np.zeros((num_metrics,))

    # Getting accuracy baseline
    with np.load(
        join(
            master_path,
            "results",
            "clustering",
            clustering_method,
            f"{dataset_name}_{vocoder_name}_{participant}_spec_c{num_classes:02d}_f{nfolds:02d}.npz",
        )
    ) as cluster_outputs:

        cluster_train = [
            cluster_outputs[f"y_train_{fold:02d}"] for fold in range(nfolds)
        ]
        histogram_counts = np.sort(
            [np.unique(x, return_counts=True)[1] for x in cluster_train]
        )
        histogram_counts = histogram_counts / np.sum(
            histogram_counts, axis=1, keepdims=True
        )

        baseline[0] = np.mean(histogram_counts[:, -1])

    correlation = np.zeros((len(participants), nfolds))
    for i_participant, participant in enumerate(participants):
        with np.load(
            join(
                master_path,
                "results",
                "clustering",
                clustering_method,
                f"{dataset_name}_{vocoder_name}_{participant}_spec_c{num_classes:02d}_f{nfolds:02d}_stats.npz",
            )
        ) as cluster_stats:
            correlation[i_participant, :] = cluster_stats["correlation"]

    corr_upperbound = np.mean(correlation)

    if num_classes > 2:
        xticks.insert(1, f"Top-{topk} Accuracy")
        xticks.insert(2, "Speech Accuracy")
        xticks.insert(3, f"Top-{topk} Speech Accuracy")

        baseline[1] = np.mean(np.sum(histogram_counts[:, -topk:], -1))

        histogram_counts_speech = histogram_counts[:, :-1]
        histogram_counts_speech = histogram_counts_speech / np.sum(
            histogram_counts_speech, axis=1, keepdims=True
        )
        baseline[2] = np.mean(histogram_counts_speech[:, -1])
        baseline[3] = np.mean(np.sum(histogram_counts_speech[:, -topk:], -1))

    # mel spectrogram baseline
    baseline[-1] = np.mean(
        [
            np.percentile(
                np.load(
                    join(
                        master_path,
                        "dataset",
                        dataset_name,
                        vocoder_name,
                        f"{participant}_spec_r_baseline.npy",
                    )
                ),
                95,
            )
            for participant in participants
        ]
    )

    x = np.arange(num_metrics)

    width_ratios = [num_metrics - 1, 1]

    fig, axs = plt.subplots(
        ncols=2,
        layout="constrained",
        figsize=(7, 5) if num_metrics == 2 else (16, 5),
        width_ratios=width_ratios,
    )

    for i_method, method in enumerate(methods):
        measure = np.array(
            [
                np.concatenate(
                    [
                        method["metrics"][participant][i_metric]
                        for participant in participants
                    ]
                )
                for i_metric in range(num_metrics)
            ]
        ).T
        y = np.nanmean(measure, axis=0)
        # y_err = np.std(measure, axis=0)
        y_err = np.abs(
            np.stack(
                (
                    np.quantile(measure, 1 / 4, axis=0),
                    np.quantile(measure, 3 / 4, axis=0),
                ),
                0,
            )
            - y[None, :]
        )
        rects = axs[0].bar(
            x[:-1] + i_method * width,
            y[:-1],
            width * 0.9,
            label=method["name"],
            linewidth=0.1,
            yerr=y_err[:, :-1],
        )
        axs[0].bar_label(rects, padding=3, fmt=lambda x: f"{x:.2%}")

        rects = axs[1].bar(
            x[-1:] + i_method * width,
            y[-1:],
            width * 0.9,
            label=method["name"],
            linewidth=0.1,
            yerr=y_err[:, -1:],
        )
        axs[1].bar_label(rects, padding=3, fmt="%.2f")
        # plt.errorbar(x + i_method * width, y, yerr=y_err, fmt="o", color="r")

    for i_metric in range(num_metrics):
        sig_bar_count = 0
        axs[0 if i_metric != num_metrics - 1 else 1].plot(
            [x[i_metric] - width / 2, x[i_metric] + (methods_size - 0.5) * width],
            [baseline[i_metric]] * 2,
            "--k",
            linewidth=0.5,
        )
        if i_metric == num_metrics - 1:
            axs[1].plot(
                [x[i_metric] - width / 2, x[i_metric] + (methods_size - 0.5) * width],
                [corr_upperbound] * 2,
                "-.k",
                linewidth=0.5,
            )
        for i_method_permute, x_method_permute in enumerate(combs):
            if ttests[i_metric, i_method_permute, 1] >= 0.05:
                continue
            elif ttests[i_metric, i_method_permute, 1] < 0.001:
                # color = "midnightblue"
                text = "***"
            elif ttests[i_metric, i_method_permute, 1] < 0.01:
                # color = "blue"
                text = "**"
            elif ttests[i_metric, i_method_permute, 1] < 0.05:
                # color = "mediumslateblue"
                text = "*"
            axs[0 if i_metric != num_metrics - 1 else 1].plot(
                [
                    x[i_metric] + combs[i_method_permute][0] * width,
                    x[i_metric] + combs[i_method_permute][1] * width,
                ],
                [1 + sig_bar_count * (0.05 if i_metric != num_metrics - 1 else 0.1)]
                * 2,
                color="k",
                linewidth=1,
            )
            axs[0 if i_metric != num_metrics - 1 else 1].text(
                x[i_metric]
                + (combs[i_method_permute][0] + combs[i_method_permute][1]) / 2 * width,
                1 + sig_bar_count * (0.05 if i_metric != num_metrics - 1 else 0.1),
                text,
                ha="center",
            )
            sig_bar_count += 1

    plt.suptitle(r"Comparing Classification Models for $SU={}$".format(num_classes))
    axs[0].set_ylabel("Percentage")
    axs[1].set_ylabel("r")
    axs[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[0].set_xticks(x[:-1] + (width * (methods_size - 1) / 2), xticks[:-1])
    axs[1].set_xticks(x[-1:] + (width * (methods_size - 1) / 2), xticks[-1:])
    axs[0].legend(
        loc="upper left", ncols=3, fontsize="xx-small" if num_metrics == 2 else "small"
    )
    axs[0].set_ylim(0, 1.2)
    axs[1].set_ylim(0, 1.4)
    plt.tight_layout()

    names = "_" + "_".join(sorted([method["name"] for method in methods]))
    fig.savefig(
        join(output_path, file_names.local_template_no_participant + names + ".png"),
        dpi=300,
    )
    plt.close()

    if len(participants) > 1:
        for i_metric in range(num_metrics):
            measure = np.nanmean(
                [
                    methods[1]["metrics"][participant][i_metric]
                    for participant in participants
                ],
                axis=1,
            )

            plt.figure()
            plt.bar(np.arange(len(participants)), measure)
            plt.xticks(np.arange(len(participants)), participants, rotation=45)
            plt.title(
                f"Transformer Encoder for different Participants in {num_classes} classes: {xticks[i_metric]}"
            )
            plt.tight_layout()
            plt.savefig(
                join(
                    output_path,
                    file_names.local_template_no_participant
                    + names
                    + "_"
                    + xticks[i_metric]
                    + ".png",
                )
            )
            plt.close()


def compare_with_reg(
    method_clf,
    method_reg,
    participants,
    nums_classes,
    nfolds,
    clustering_method,
    results_dir,
    dataset_name,
    vocoder_name,
    output_path,
):

    for num_classes in nums_classes:
        method_clf[num_classes] = []
        for participant in participants:
            file_names = utils.names.Names(
                join(results_dir, method_clf["dir"]),
                dataset_name,
                vocoder_name,
                participant,
                nfolds,
                num_classes,
                clustering_method,
            )
            file_names.update(fold="all")
            stats_clf = np.load(file_names.stats + ".npy", allow_pickle=True).item()

            method_clf[num_classes].append(np.nanmean(stats_clf["test_melSpec_corr"]))
    file_names = utils.names.Names(
        join(results_dir, method_reg["dir"]),
        dataset_name,
        vocoder_name,
        participant,
        nfolds,
        0,
        clustering_method,
    )
    file_names.update(fold="all")
    stats_reg = np.load(file_names.output_path + "\\results.pck", allow_pickle=True)

    method_reg[0] = [
        np.nanmean(stats_reg[x]["correlation_coef_per_bin"])
        for x in sorted(stats_reg.keys())
    ]

    correlation = np.zeros((len(nums_classes), len(participants), nfolds))
    for i_nums_classes, num_classes in enumerate(nums_classes):
        for i_participant, participant in enumerate(participants):
            with np.load(
                join(
                    master_path,
                    "results",
                    "clustering",
                    clustering_method,
                    f"{dataset_name}_{vocoder_name}_{participant}_spec_c{num_classes:02d}_f{nfolds:02d}_stats.npz",
                )
            ) as cluster_stats:
                correlation[i_nums_classes, i_participant, :] = cluster_stats[
                    "correlation"
                ]

    corr_upperbound = np.mean(correlation, axis=-1)

    baseline = np.array(
        [
            np.percentile(
                np.load(
                    join(
                        master_path,
                        "dataset",
                        dataset_name,
                        vocoder_name,
                        f"{participant}_spec_r_baseline.npy",
                    )
                ),
                95,
            )
            for participant in participants
        ]
    )

    measure = np.stack(
        [method_clf[num_classes] for num_classes in nums_classes] + [method_reg[0]]
    )

    num_models = len(nums_classes) + 1

    fig, ax = plt.subplots(
        layout="constrained",
        figsize=(10, 5),
    )

    x = np.arange(len(participants))
    width = 0.8 / (num_models + 1)

    names = [f"TE-{num_classes}" for num_classes in nums_classes] + ["TE-reg"]

    for i_num_models in range(num_models):
        rects = ax.bar(
            x + width * i_num_models,
            measure[i_num_models],
            width * 0.9,
            linewidth=0.1,
            label=names[i_num_models],
        )
        # ax.bar_label(rects, label_type="center", padding=3, fmt="%.2f")

    for i_participant, participant in enumerate(participants):
        ax.plot(
            [
                x[i_participant] - width / 2,
                x[i_participant] + (num_models - 0.5) * width,
            ],
            [baseline[i_participant]] * 2,
            "--k",
            linewidth=0.5,
        )
        for i_nums_classes, num_classes in enumerate(nums_classes):
            ax.plot(
                [
                    x[i_participant] + (i_nums_classes - 0.5) * width,
                    x[i_participant] + (i_nums_classes + 0.5) * width,
                ],
                [corr_upperbound[i_nums_classes, i_participant]] * 2,
                "-.k",
                linewidth=0.5,
            )

    plt.suptitle(r"Comparing Classification Models with a Regression model")
    ax.set_ylabel("r")
    ax.legend(loc="upper left", ncols=num_models, fontsize="x-small")
    ax.set_ylim(-0.05, 1)
    ax.set_xticks(x + (width * num_models / 2), participants, rotation=45)
    plt.tight_layout()
    names = "_" + "_".join(sorted(names))
    fig.savefig(
        join(
            output_path,
            file_names.local_template_no_participant_no_num_classes + names + ".png",
        ),
        dpi=300,
    )
    plt.close()


def main():

    participants = [f"sub-{i:02d}" for i in range(1, 11)]
    nfolds = 10
    dataset_name = "Word"
    clustering_method = "kmeans"
    topk = 3

    vocoder_name = "VocGAN"  # "Griffin_Lim"

    output_path = join(master_path, "results", "comparison_SA", "classification")
    makedirs(output_path, exist_ok=True)

    results_dir = join(master_path, "results", "classification")
    method1 = {}
    method1["dir"] = join("1", "LDA")
    method1["name"] = "LDA"

    method2 = {}
    method2["dir"] = join("40", "Transformer_Encoder")
    method2["name"] = "TE"

    method3 = {}
    method3["dir"] = join("40", "RNN")
    method3["name"] = "RNN"

    methods = (method1, method2, method3)  #

    for num_classes in (2, 5, 20):
        compare(
            methods,
            participants,
            num_classes,
            nfolds,
            clustering_method,
            results_dir,
            dataset_name,
            vocoder_name,
            topk,
            output_path,
        )

    method4 = {}
    method4["dir"] = join("40", "Transformer_Encoder_halfdata")
    method4["name"] = "TE-half"

    methods = (method2, method4)

    for num_classes in (2, 5, 20):
        compare(
            methods,
            ["sub-06"],
            num_classes,
            nfolds,
            clustering_method,
            results_dir,
            dataset_name,
            vocoder_name,
            topk,
            output_path,
        )

    method_reg = {}
    method_reg["dir"] = join("..", "regression", "40", "Transformer_Encoder")
    method_reg["name"] = "TE-reg"

    method_clf = method2
    nums_classes = (2, 5, 20)
    compare_with_reg(
        method_clf,
        method_reg,
        participants,
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
