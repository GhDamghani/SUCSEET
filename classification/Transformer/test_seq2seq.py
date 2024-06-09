def save_accuracy_plot_seq2seq(
    test_acc, test_acc_speech, test_topk_acc, test_topk_acc_speech, config
):
    max_plot = 1.0
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.title("General Accuracy Plot")
    plt.bar(
        range(config.window_size),
        test_topk_acc,
        color="forestgreen",
        label=f"Top-{config.topk} Accuracy",
    )
    plt.bar(range(config.window_size), test_acc, color="royalblue", label="Accuracy")
    plt.ylim(0, max_plot)
    plt.xlabel("Window Timepoint Index")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Speech Accuracy Plot")
    plt.bar(
        range(config.window_size),
        test_topk_acc_speech,
        color="forestgreen",
        label=f"Top-{config.topk} Accuracy",
    )
    plt.bar(
        range(config.window_size), test_acc_speech, color="royalblue", label="Accuracy"
    )
    plt.ylim(0, max_plot)
    plt.xlabel("Window Timepoint Index")

    plt.legend()
    plt.tight_layout()
    plt.savefig(join(config.output_path, "accuracy_plot.png"))


def main_seq2seq(miniconfig):
    import config

    config0 = config.Config()
    config0.participant = miniconfig["participant"]
    config0.num_classes = miniconfig["num_classes"]
    config0.topk = 3
    stats = {}

    y_test = [[] for _ in range(config0.window_size)]
    y_pred_train = [[] for _ in range(config0.window_size)]
    y_pred_test = [[] for _ in range(config0.window_size)]

    y_pred_test_labels = [[] for _ in range(config0.window_size)]

    centred_melSpec = [[] for _ in range(config0.window_size)]

    train_corrects = [0 for _ in range(config0.window_size)]
    test_corrects = [0 for _ in range(config0.window_size)]

    train_corrects_speech = [0 for _ in range(config0.window_size)]
    test_corrects_speech = [0 for _ in range(config0.window_size)]

    train_topk_corrects = [0 for _ in range(config0.window_size)]
    test_topk_corrects = [0 for _ in range(config0.window_size)]
    train_total = [0 for _ in range(config0.window_size)]
    test_total = [0 for _ in range(config0.window_size)]

    train_topk_corrects_speech = [0 for _ in range(config0.window_size)]
    test_topk_corrects_speech = [0 for _ in range(config0.window_size)]
    train_total_speech = [0 for _ in range(config0.window_size)]
    test_total_speech = [0 for _ in range(config0.window_size)]

    for i in range(config0.nfolds):
        config0.fold = i
        config0.proc()
        y_pred_train0, y_train0, y_pred_test0, y_test0, model_summary = test_results(
            config0
        )
        y_pred_test_labels0 = np.argmax(y_pred_test0, axis=-1)
        centred_melSpec0 = np.stack(
            tuple(config0.melSpec_centers[x] for x in y_pred_test_labels0), axis=0
        )
        train_corrects0 = np.stack(
            [
                accuracy_score(
                    y_train0[:, i],
                    np.argmax(y_pred_train0[:, i, :], axis=-1),
                    normalize=False,
                )
                for i in range(config0.window_size)
            ],
            axis=0,
        )

        test_corrects0 = np.stack(
            [
                accuracy_score(
                    y_test0[:, i],
                    np.argmax(y_pred_test0[:, i, :], axis=-1),
                    normalize=False,
                )
                for i in range(config0.window_size)
            ],
            axis=0,
        )
        if config0.num_classes > 2:
            train_topk_corrects0 = np.stack(
                [
                    top_k_accuracy_score(
                        y_train0[:, i],
                        y_pred_train0[:, i],
                        k=config0.topk,
                        normalize=False,
                    )
                    for i in range(config0.window_size)
                ],
                axis=0,
            )

            test_topk_corrects0 = np.stack(
                [
                    top_k_accuracy_score(
                        y_test0[:, i],
                        y_pred_test0[:, i],
                        k=config0.topk,
                        normalize=False,
                    )
                    for i in range(config0.window_size)
                ],
                axis=0,
            )
        for i in range(config0.window_size):
            train_speech_mask0 = y_train0[:, i] != 0
            y_train_speech0 = y_train0[:, i][train_speech_mask0] - 1
            y_pred_train_speech0 = y_pred_train0[:, i][train_speech_mask0, 1:]
            train_corrects_speech0 = accuracy_score(
                y_train_speech0,
                np.argmax(y_pred_train_speech0, axis=-1),
                normalize=False,
            )
            if config0.num_classes > 2:
                train_topk_corrects_speech0 = top_k_accuracy_score(
                    y_train_speech0,
                    y_pred_train_speech0,
                    k=config0.topk,
                    normalize=False,
                )
            test_speech_mask0 = y_test0[:, i] != 0

            y_test_speech0 = y_test0[:, i][test_speech_mask0] - 1
            y_pred_test_speech0 = y_pred_test0[:, i][test_speech_mask0, 1:]

            test_corrects_speech0 = accuracy_score(
                y_test_speech0, np.argmax(y_pred_test_speech0, axis=1), normalize=False
            )
            if config0.num_classes > 2:
                test_topk_corrects_speech0 = top_k_accuracy_score(
                    y_test_speech0, y_pred_test_speech0, k=config0.topk, normalize=False
                )

            y_test[i].append(y_test0[:, i])
            y_pred_train[i].append(y_pred_train0[:, i])
            y_pred_test[i].append(y_pred_test0[:, i])

            y_pred_test_labels[i].append(y_pred_test_labels0[:, i])

            centred_melSpec[i].append(centred_melSpec0[:, i])

            train_corrects[i] += train_corrects0[i]
            test_corrects[i] += test_corrects0[i]
            train_corrects_speech[i] += train_corrects_speech0
            test_corrects_speech[i] += test_corrects_speech0
            if config0.num_classes > 2:
                train_topk_corrects[i] += train_topk_corrects0[i]
                test_topk_corrects[i] += test_topk_corrects0[i]
            train_total[i] += len(y_train0)
            test_total[i] += len(y_test0)

            if config0.num_classes > 2:
                train_topk_corrects_speech[i] += train_topk_corrects_speech0
                test_topk_corrects_speech[i] += test_topk_corrects_speech0
            train_total_speech[i] += len(y_train_speech0)
            test_total_speech[i] += len(y_test_speech0)
    train_acc = np.array(train_corrects) / np.array(train_total)

    train_total = np.array(train_total)
    test_corrects = np.array(test_corrects)
    test_total = np.array(test_total)
    train_corrects_speech = np.array(train_corrects_speech)
    train_total_speech = np.array(train_total_speech)
    test_corrects_speech = np.array(test_corrects_speech)
    test_total_speech = np.array(test_total_speech)

    test_acc = test_corrects / test_total
    train_acc_speech = train_corrects_speech / train_total_speech
    test_acc_speech = test_corrects_speech / test_total_speech
    if config0.num_classes > 2:
        train_topk_corrects = np.array(train_topk_corrects)
        test_topk_corrects = np.array(test_topk_corrects)
        train_topk_corrects_speech = np.array(train_topk_corrects_speech)
        test_topk_corrects_speech = np.array(test_topk_corrects_speech)
        train_topk_acc = train_topk_corrects / train_total
        test_topk_acc = test_topk_corrects / test_total
        train_topk_acc_speech = train_topk_corrects_speech / train_total_speech
        test_topk_acc_speech = test_topk_corrects_speech / test_total_speech

    y_test = np.stack([np.concatenate(x) for x in y_test])
    y_pred_test_labels = np.stack([np.concatenate(x) for x in y_pred_test_labels])
    centred_melSpec = np.stack([np.concatenate(x) for x in centred_melSpec])

    stats["train_acc"] = train_acc
    stats["test_acc"] = test_acc
    stats["train_topk_acc"] = train_topk_acc
    stats["test_topk_acc"] = test_topk_acc
    stats["train_acc_speech"] = train_acc_speech
    stats["test_acc_speech"] = test_acc_speech
    stats["train_topk_acc_speech"] = train_topk_acc_speech
    stats["test_topk_acc_speech"] = test_topk_acc_speech
    np.savez_compressed(config0.output_file_name + "_stats", stats)

    with open(
        config0.output_file_name + "_model_summary.txt", "w", encoding="utf-8"
    ) as f:
        f.write(model_summary)

    save_accuracy_plot_seq2seq(
        test_acc, test_acc_speech, test_topk_acc, test_topk_acc_speech, config0
    )
