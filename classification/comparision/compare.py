import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os import makedirs

master_path = join("..", "..")

import sys

sys.path.append(master_path)

import utils


def main():
    num_classes = 5
    participant = "sub-06"  # "p07_ses1_sentences"
    nfolds = 10
    dataset_name = "Word"
    clustering_method = "kmeans"
    topk = 3

    vocoder_name = "VocGAN"  # "Griffin_Lim"

    output_path = join(master_path, "results", "comparison")
    makedirs(output_path, exist_ok=True)

    results_dir = join(master_path, "results", "classification")
    method1 = {}
    method1["dir"] = join("1", "LDA")
    method1["name"] = "LDA"

    method2 = {}
    method2["dir"] = join("40", "Transformer_Encoder")
    method2["name"] = "Transformer_Encoder"

    method3 = {}
    method3["dir"] = join("40", "RNN")
    method3["name"] = "RNN"

    methods = (method1, method2, method3)  #
    methods_size = len(methods)

    for i_method, method in enumerate(methods):
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
        method["stats"] = np.load(file_names.stats + ".npy", allow_pickle=True).item()

    width = 0.5 / methods_size

    xticks = ["Accuracy", "MelSpec Corr"]
    num_measures = len(xticks)
    if num_classes > 2:
        num_measures += 3
    x = np.arange(num_measures)

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 5))
    for i_method, method in enumerate(methods):
        test_accuracy = np.mean(
            np.array(method["stats"]["test_corrects"])
            / np.array(method["stats"]["test_total"])
        )

        test_melSpec_corr = np.mean(
            np.concatenate(method["stats"]["test_melSpec_corr"])
        )
        measure = [test_accuracy, test_melSpec_corr]

        if num_classes > 2:
            test_topk_accuracy = np.mean(
                np.array(method["stats"]["test_topk_corrects"])
                / np.array(method["stats"]["test_total"])
            )
            test_speech_accuracy = np.mean(
                np.array(method["stats"]["test_corrects_speech"])
                / np.array(method["stats"]["test_total_speech"])
            )
            test_topk_speech_accuracy = np.mean(
                np.array(method["stats"]["test_topk_corrects_speech"])
                / np.array(method["stats"]["test_total_speech"])
            )
            measure.insert(1, test_topk_accuracy)
            measure.insert(2, test_speech_accuracy)
            measure.insert(3, test_topk_speech_accuracy)

        rects = plt.bar(x + i_method * width, measure, width, label=method["name"])
        ax.bar_label(rects, padding=3, fmt="%.2f")

    if num_classes > 2:
        xticks.insert(1, f"Top-{topk} Accuracy")
        xticks.insert(2, "Speech Accuracy")
        xticks.insert(3, f"Top-{topk} Speech Accuracy")
    ax.set_ylabel("Percentage")
    ax.set_title("Comparing Classification Models")
    ax.set_xticks(x + (width * (methods_size - 1) / 2), xticks)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 1.1)

    names = "_" + "_".join(sorted([method["name"] for method in methods]))
    fig.savefig(join(output_path, file_names.local + names + ".png"), dpi=300)

    # plt.show()


if __name__ == "__main__":
    main()
