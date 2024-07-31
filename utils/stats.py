from sklearn.metrics import ConfusionMatrixDisplay, top_k_accuracy_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def save_confusion_matrix(
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
    file_name,
    extra_title=None,
):

    conf_mat_title = f"{participant}, {num_classes} classes, {clf_name}\n Vocoder: {vocoder_name} kfold iteration: {fold}\nAccuracy: {test_acc:.3%}"
    if num_classes > 2:
        conf_mat_title += f" (Speech: {test_speech_acc:.3%})\nTop-{topk} Accuracy: {test_topk_acc:.3%} (Speech: {test_topk_speech_acc:.3%})"
    if extra_title:
        conf_mat_title += extra_title
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_test_labels, normalize="true", text_kw={"fontsize": 12}
    )
    plt.gcf().set_size_inches(7, 7)
    plt.title(conf_mat_title)
    plt.savefig(file_name)
    plt.close()


def save_histograms(
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
    file_name,
    extra_title=None,
):
    hist_suptitle = f"{participant}, {num_classes} classes, {clf_name}\n Vocoder: {vocoder_name}\nkfold iteration: {fold} Accuracy: {test_acc:.3%} (train: {train_acc:.3%})"
    if num_classes > 2:
        hist_suptitle += f" (Speech: {test_speech_acc:.3%})\nTop-{topk} Accuracy: {test_topk_acc:.3%} (Speech: {test_topk_speech_acc:.3%})"
    if extra_title:
        hist_suptitle += extra_title

    labels = np.arange(num_classes)

    pred_labels_uniques, pred_counts = np.unique(y_pred_test_labels, return_counts=True)
    y_labels_uniques, y_counts = np.unique(y_test.astype(int), return_counts=True)

    y_hist = np.zeros(num_classes)
    y_hist[y_labels_uniques] = y_counts
    pred_hist = np.zeros(num_classes)
    pred_hist[pred_labels_uniques] = pred_counts
    total = np.sum(y_hist)

    pred_hist = pred_hist / total
    y_hist = y_hist / total

    # print(labels)
    # print(y_hist)
    # print(pred_hist)

    max_plot = max(pred_hist.max(), y_hist.max()) * 1.1

    # print(f"Total y Counts: {total}")

    plt.figure(figsize=(6, 4))

    plt.stem(
        labels, pred_hist, linefmt="C0-", markerfmt="o", basefmt="k-", label="Predicted"
    )
    plt.stem(
        labels + 0.1, y_hist, linefmt="C1-", markerfmt="o", basefmt="k-", label="True"
    )
    plt.xticks(range(num_classes))
    plt.xlabel("Labels")
    plt.ylabel("Freq")
    plt.ylim(0, max_plot)
    plt.title("Histogram of True and Predicted Output for test data")
    plt.legend(loc="upper right")

    plt.suptitle(hist_suptitle)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def pearson_corr(x, y, axis=0):

    x_mean = np.mean(x, axis=axis, keepdims=True)
    y_mean = np.mean(y, axis=axis, keepdims=True)

    nominator = np.sum((x - x_mean) * (y - y_mean), axis=axis)
    denominator1 = np.sqrt(np.sum(np.square(x - x_mean), axis=axis))
    denominator2 = np.sqrt(np.sum(np.square(y - y_mean), axis=axis))
    denominator = denominator1 * denominator2

    rho = nominator / denominator
    return rho


from torch.nn.functional import cosine_similarity
import torch


def calculate_pearsonr(logits, y, axis=1):
    logits = torch.from_numpy(logits)
    y = torch.from_numpy(y)
    pearson_r = cosine_similarity(
        y - torch.mean(y, dim=axis).unsqueeze(axis),
        logits - torch.mean(logits, dim=axis).unsqueeze(axis),
        dim=axis,
    )
    return pearson_r.numpy()


def save_stats(
    results,
    melSpecs_train,
    melSpecs_test,
    window_size,
    nfolds,
    num_classes,
    topk,
    output_file_name_stats,
):

    train_corrects = []
    test_corrects = []

    train_corrects_speech = []
    test_corrects_speech = []

    if num_classes > 2:
        train_topk_corrects = []
        test_topk_corrects = []
    train_total = []
    test_total = []

    if num_classes > 2:
        train_topk_corrects_speech = []
        test_topk_corrects_speech = []
    train_total_speech = []
    test_total_speech = []

    test_melSpec_corr = np.zeros((nfolds,))

    for fold in range(nfolds):
        y_train = results["y_train"][fold]
        y_test = results["y_test"][fold]
        y_pred_train = results["y_pred_train"][fold]
        y_pred_test = results["y_pred_test"][fold]
        centred_melSpec_test = results["centred_melSpec_test"][fold]

        y_pred_train_labels = np.argmax(y_pred_train, axis=1)
        y_pred_test_labels = np.argmax(y_pred_test, axis=1)

        train_corrects.append(
            accuracy_score(y_train, y_pred_train_labels, normalize=False)
        )
        test_corrects.append(
            accuracy_score(y_test, y_pred_test_labels, normalize=False)
        )

        train_total.append(len(y_train))
        test_total.append(len(y_test))

        if num_classes > 2:

            train_topk_corrects.append(
                top_k_accuracy_score(
                    y_train,
                    y_pred_train,
                    k=topk,
                    normalize=False,
                    labels=range(num_classes),
                )
            )
            test_topk_corrects.append(
                top_k_accuracy_score(
                    y_test,
                    y_pred_test,
                    k=topk,
                    normalize=False,
                    labels=range(num_classes),
                )
            )

        train_speech_mask = y_train != 0
        test_speech_mask = y_test != 0

        y_train_speech = y_train[train_speech_mask] - 1
        y_pred_train_speech = y_pred_train[train_speech_mask, 1:]
        y_test_speech = y_test[test_speech_mask] - 1
        y_pred_test_speech = y_pred_test[test_speech_mask, 1:]

        train_corrects_speech.append(
            accuracy_score(
                y_train_speech, np.argmax(y_pred_train_speech, axis=1), normalize=False
            )
        )
        test_corrects_speech.append(
            accuracy_score(
                y_test_speech, np.argmax(y_pred_test_speech, axis=1), normalize=False
            )
        )

        train_total_speech.append(len(y_train_speech))
        test_total_speech.append(len(y_test_speech))

        if num_classes > 2:
            train_topk_corrects_speech.append(
                top_k_accuracy_score(
                    y_train_speech,
                    y_pred_train_speech,
                    k=topk,
                    normalize=False,
                    labels=range(num_classes - 1),
                )
            )
            test_topk_corrects_speech.append(
                top_k_accuracy_score(
                    y_test_speech,
                    y_pred_test_speech,
                    k=topk,
                    normalize=False,
                    labels=range(num_classes - 1),
                )
            )

        test_melSpec_corr[fold] = np.nanmean(
            pearson_corr(
                centred_melSpec_test, melSpecs_test[fold][window_size - 1 :], axis=0
            ),
        )

    stats = {
        "train_corrects": train_corrects,
        "test_corrects": test_corrects,
        "train_corrects_speech": train_corrects_speech,
        "test_corrects_speech": test_corrects_speech,
        "train_total": train_total,
        "test_total": test_total,
        "train_total_speech": train_total_speech,
        "test_total_speech": test_total_speech,
        "test_melSpec_corr": test_melSpec_corr,
    }
    if num_classes > 2:
        stats["train_topk_corrects"] = train_topk_corrects
        stats["test_topk_corrects"] = test_topk_corrects
        stats["train_topk_corrects_speech"] = train_topk_corrects_speech
        stats["test_topk_corrects_speech"] = test_topk_corrects_speech

    np.save(output_file_name_stats, stats)


def save_stats_reg(
    results,
    melSpecs_train,
    melSpecs_test,
    window_size,
    nfolds,
    output_file_name_stats,
):

    train_total = []
    test_total = []

    test_melSpec_corr = []

    for fold in range(nfolds):
        y_train = results["y_train"][fold]
        y_test = results["y_test"][fold]
        y_pred_train = results["y_pred_train"][fold]
        y_pred_test = results["y_pred_test"][fold]

        shift = 3
        y_pred_test_0 = y_pred_test[0, :]
        y_pred_test_rest = y_pred_test[1::shift, -shift:].reshape(
            -1, y_pred_test.shape[2]
        )

        y_pred_test = np.concatenate((y_pred_test_0, y_pred_test_rest), axis=0)

        train_total.append(len(y_train))
        test_total.append(len(y_test))

        test_melSpec_corr.append(
            np.nanmean(
                pearson_corr(
                    y_pred_test[: melSpecs_test[fold].shape[0]],
                    melSpecs_test[fold],
                    axis=0,
                )
            )
        )

    stats = {
        "train_total": train_total,
        "test_total": test_total,
        "test_melSpec_corr": test_melSpec_corr,
    }

    np.save(output_file_name_stats, stats)


def save_stats_summary(file_names, num_classes):
    fold = "all"
    file_names.update(fold)
    stats = np.load(file_names.stats + ".npy", allow_pickle=True).item()

    train_accuracy = np.sum(stats["train_corrects"]) / np.sum(stats["train_total"])
    test_accuracy = np.sum(stats["test_corrects"]) / np.sum(stats["test_total"])
    train_speech_accuracy = np.sum(stats["train_corrects_speech"]) / np.sum(
        stats["train_total_speech"]
    )
    test_speech_accuracy = np.sum(stats["test_corrects_speech"]) / np.sum(
        stats["test_total_speech"]
    )
    test_melSpec_corr = np.nanmean(stats["test_melSpec_corr"])
    if num_classes > 2:
        train_topk_accuracy = np.sum(stats["train_topk_corrects"]) / np.sum(
            stats["train_total"]
        )
        test_topk_accuracy = np.sum(stats["test_topk_corrects"]) / np.sum(
            stats["test_total"]
        )
    else:
        train_topk_accuracy = None
        test_topk_accuracy = None
    with open(file_names.stats + ".txt", "w") as f:
        f.write(f"train_accuracy: {train_accuracy}\n")
        f.write(f"test_accuracy: {test_accuracy}\n")
        f.write(f"train_speech_accuracy: {train_speech_accuracy}\n")
        f.write(f"test_speech_accuracy: {test_speech_accuracy}\n")
        f.write(f"test_melSpec_corr: {test_melSpec_corr}\n")
        if num_classes > 2:
            f.write(f"train_topk_accuracy: {train_topk_accuracy}\n")
            f.write(f"test_topk_accuracy: {test_topk_accuracy}\n")
