import torch
from model import SpeechDecodingModel_clf
import loss_metrics
import numpy as np
from os.path import join
import scipy
import sys
from sklearn.model_selection import KFold

master_path = "../.."
sys.path.append(master_path)


import utils
from utils.model_trainer.trainer import Trainer

from utils.vocoders.Griffin_Lim import createAudio
from utils.vocoders.VocGAN import StreamingVocGan


# def save_confusion_matrix_speech(
#     y_test,
#     y_pred_test,
#     train_acc,
#     test_acc,
#     config,
#     isfolds=False,
# ):
#     if isfolds is False:
#         conf_mat_title = f"{config.participant}, {config.num_classes} classes, {config.clf_name}\n Vocoder: {config.vocoder_name}\nkfold iteration: {config.fold} Accuracy: {test_acc:.3%} (train: {train_acc:.3%})"
#     else:
#         conf_mat_title = f"{config.participant}, {config.num_classes} classes, {config.clf_name}\n Vocoder: {config.vocoder_name}\nAll Folds, Accuracy: {test_acc:.3%} (train: {train_acc:.3%})"
#     output_file_name = config.output_file_name + f"_confusion_speech.png"
#     ConfusionMatrixDisplay.from_predictions(
#         y_test, y_pred_test, normalize="true", text_kw={"fontsize": 12}
#     )
#     plt.gcf().set_size_inches(7, 7)
#     plt.title(conf_mat_title)
#     plt.savefig(output_file_name)


def save_reconstruction(centred_melSpec, config):
    output_file_name = config.output_file_name + ".wav"
    if config.vocoder_name == "VocGAN":
        model_path = join(
            master_path, "utils", "vocoders", "VocGAN", "vctk_pretrained_model_3180.pt"
        )
        VocGAN = StreamingVocGan(model_path, is_streaming=False)
        centred_melSpec = torch.tensor(np.transpose(centred_melSpec).astype(np.float32))
        center_audio, _ = VocGAN.mel_spectrogram_to_waveform(
            mel_spectrogram=centred_melSpec
        )
        StreamingVocGan.save(
            waveform=center_audio,
            file_path=output_file_name,
        )
    elif config.vocoder_name == "Griffin_Lim":
        audiosr = 16000
        center_audio = createAudio(centred_melSpec, audiosr)
        scipy.io.wavfile.write(
            join(config.output_path, output_file_name),
            int(audiosr),
            center_audio,
        )


def get_test_results(config):

    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)

    dataset = iter(
        utils.data.WindowedData(
            config.feat,
            config.cluster,
            window_size=config.window_size,
            num_folds=config.nfolds,
            output_indices=(-1,),
        )
    )
    for i in range(config.fold + 1):
        train_dataset, val_dataset = next(dataset)

    model = SpeechDecodingModel_clf(
        config.d_model,
        config.num_heads,
        config.dim_feedforward,
        config.num_layers,
        config.num_classes,
        config.window_size,
        config.feat_size,
        config.output_indices,
        config.dropout,
    )
    model.to(config.DEVICE)
    model.load_state_dict(torch.load(config.file_names.model))

    loss, _ = loss_metrics.get_loss(train_dataset, config.num_classes)

    trainer = Trainer(
        model,
        loss,
        config.metrics,
        val_dataset,
        config.model_task,
        batch_size=config.BATCH_SIZE * 4,
        device=config.DEVICE,
    )
    y_pred_test, y_test, _ = trainer.validate(return_data=True)

    trainer.val_dataset = train_dataset
    y_pred_train, y_train, _ = trainer.validate(return_data=True)

    print(f"Testing {config.file_names.file} completed")
    return y_pred_train, y_train, y_pred_test, y_test


def save_test_results(config):
    y_train = []
    y_test = []

    y_pred_train = []
    y_pred_test = []

    centred_melSpec_train = []
    centred_melSpec_test = []

    for i in range(config.nfolds):
        config.fold = i
        config.proc()
        y_pred_train0, y_train0, y_pred_test0, y_test0 = get_test_results(config)

        y_pred_train0 = np.squeeze(y_pred_train0)
        y_train0 = np.squeeze(y_train0)
        y_pred_test0 = np.squeeze(y_pred_test0)
        y_test0 = np.squeeze(y_test0)

        y_pred_train_labels0 = np.argmax(y_pred_train0, axis=1)
        y_pred_test_labels0 = np.argmax(y_pred_test0, axis=1)

        y_pred_train.append(np.squeeze(y_pred_train0))
        y_train.append(np.squeeze(y_train0))
        y_pred_test.append(np.squeeze(y_pred_test0))
        y_test.append(np.squeeze(y_test0))
        centred_melSpec_train0 = np.stack(
            tuple(config.melSpec_centers[x] for x in y_pred_train_labels0), axis=0
        )
        centred_melSpec_test0 = np.stack(
            tuple(config.melSpec_centers[x] for x in y_pred_test_labels0), axis=0
        )
        centred_melSpec_train.append(centred_melSpec_train0)
        centred_melSpec_test.append(centred_melSpec_test0)
    data = {
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "centred_melSpec_train": centred_melSpec_train,
        "centred_melSpec_test": centred_melSpec_test,
    }
    config.fold = "all"
    config.update()
    np.save(config.file_names.results, data, allow_pickle=True)


def save_stats(config):
    config.fold = "all"
    config.update()
    results = np.load(config.file_names.results + ".npy", allow_pickle=True).item()

    melSpecs_train = []
    melSpecs_test = []
    config.fold = 0
    config.proc()
    for fold, (train_index, test_index) in enumerate(
        KFold(n_splits=config.nfolds, shuffle=False).split(config.feat)
    ):
        config.fold = fold
        config.proc()
        melSpecs_train.append(config.melSpec[train_index])
        melSpecs_test.append(config.melSpec[test_index])

    config.fold = "all"
    config.update()
    utils.stats.save_stats(
        results,
        melSpecs_train,
        melSpecs_test,
        config.window_size,
        config.nfolds,
        config.num_classes,
        config.topk,
        config.file_names.stats,
    )


def save_post_results(config):
    config.fold = "all"
    config.update()
    results = np.load(config.file_names.results + ".npy", allow_pickle=True).item()
    stats = np.load(config.file_names.stats + ".npy", allow_pickle=True).item()
    for i in range(config.nfolds):
        config.fold = i
        config.update()
        y_test = results["y_test"][i]
        y_pred_test = results["y_pred_test"][i]
        train_acc = stats["train_corrects"][i] / stats["train_total"][i]
        test_acc = stats["test_corrects"][i] / stats["test_total"][i]
        test_speech_acc = (
            stats["test_corrects_speech"][i] / stats["test_total_speech"][i]
        )
        if config.num_classes > 2:
            test_topk_acc = stats["test_topk_corrects"][i] / stats["test_total"][i]
            test_topk_speech_acc = (
                stats["test_topk_corrects_speech"][i] / stats["test_total_speech"][i]
            )
        else:
            test_topk_acc = None
            test_topk_speech_acc = None
        y_pred_test_labels = np.argmax(y_pred_test, axis=1)
        centred_melSpec = results["centred_melSpec_test"][i]
        utils.stats.save_confusion_matrix(
            y_test,
            y_pred_test_labels,
            test_acc,
            test_speech_acc,
            test_topk_acc,
            test_topk_speech_acc,
            config.participant,
            config.num_classes,
            config.clf_name,
            config.vocoder_name,
            config.fold,
            config.topk,
            config.file_names.confusion,
        )
        utils.stats.save_histograms(
            y_test,
            y_pred_test_labels,
            train_acc,
            test_acc,
            test_speech_acc,
            test_topk_acc,
            test_topk_speech_acc,
            config.participant,
            config.num_classes,
            config.clf_name,
            config.vocoder_name,
            config.fold,
            config.topk,
            config.file_names.histogram,
        )
        # save_reconstruction(centred_melSpec, config)
        utils.stats.save_stats_summary(config.file_names, config.num_classes)


def main():
    import config
    from multiprocessing import Pool
    from itertools import product

    participants = ["sub-06"]  # [f"sub-{i:02d}" for i in range(1, 11)]
    nums_classes = (2, 5)

    miniconfigs = [
        {"participant": participant, "num_classes": num_classes}
        for participant, num_classes in product(participants, nums_classes)
    ]

    for miniconfig in miniconfigs:
        config0 = config.Config()
        config0.participant = miniconfig["participant"]
        config0.num_classes = miniconfig["num_classes"]
        save_test_results(config0)
        save_stats(config0)
        save_post_results(config0)

    # miniconfigs[0]["fold"] = folds[0]
    # main(miniconfigs[0], isfolds=True)
    # main_seq2seq(miniconfigs[0])
    # with Pool() as pool:
    #     pool.map(main_seq2seq, miniconfigs)
    # for miniconfigs0 in miniconfigs:
    #     main_seq2seq(miniconfigs0)


if __name__ == "__main__":
    main()
