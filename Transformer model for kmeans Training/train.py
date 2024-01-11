import numpy as np
from os.path import join, exists
from os import remove, listdir
import joblib
from sklearn.utils import gen_batches
from multiprocessing import Process, freeze_support, Pipe
from model import SpeechDecodingModel
import torch

from tqdm import tqdm
from datautils import train_data_gen_proc, train_test_label
from icecream import ic
from train_step import train_model, evaluate_model
from checkpoint import ModelState
from time import sleep
import logging
from functools import partial


def logger_fcn(s, logger, right=None):
    console_width = 80
    s_len = len(s)
    if s_len == 1:
        return logger_fcn(console_width * s, logger)
    else:
        if right is None:
            return logger.info(f"| {s:<{console_width}} |")
        elif len(right) == 1:
            s = f"{(console_width//2 - s_len // 2 - 1) * right} {s} {(console_width//2 - (s_len - (s_len // 2)) - 1) * right}"
            return logger_fcn(s, logger)
        else:
            s = f'{s}{(console_width-s_len-len(right))*" "}{right}'
            return logger_fcn(s, logger)


def autosave(time, sender):
    while True:
        sleep(time)
        sender.send(True)


class Namespace:
    pass


if __name__ == "__main__":
    freeze_support()

    #### Initialize ####
    # checkpointfile = "checkpoint_2024-01-10_15-27-39"
    checkpointfile = None
    autosave_seconds = 5 * 60

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    #### Data and Hyperparameters (I) ####
    epochs = 50
    batch_size = 16

    kmeans_folder = join("..", "Kmeans of sounds")
    path_input = join(kmeans_folder, "features")
    participant = "sub-06"

    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))
    kmeans = joblib.load(join(kmeans_folder, "kmeans.joblib"))
    num_classes = 3

    no_samples = melSpec.shape[0]
    window_width = 96
    train_indices, test_indices = train_test_label(
        no_samples, window_width, p_sample=1.0
    )
    no_data_train = len(train_indices)
    no_data_test = len(test_indices)
    train_batches_indices = list(gen_batches(no_data_train, batch_size))
    test_batches_indices = list(gen_batches(no_data_test, batch_size * 2))
    no_train_batches = len(train_batches_indices)
    no_test_batches = len(test_batches_indices)
    np.random.seed(26)  # reproducibility
    train_shuffle_seeds = np.random.randint(1, 100000, epochs)
    train_data_path = "train_data"
    test_data_path = "test_data"

    #### Model and Hyperparameters (II) ####

    d_model = 128
    num_heads = 4
    dim_feedforward = 256
    num_layers = 1
    dropout = 0.1

    model = SpeechDecodingModel(
        d_model, num_classes, num_heads, dropout, num_layers, dim_feedforward
    ).to(device)
    print(model)
    print(
        "Total number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    lr = 1e-5
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20.0, gamma=0.95)
    lossfn = torch.nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.1)
    init_class = 1

    #### Logger ####
    logging_file = "train.log"
    filemode = "w" if checkpointfile is None else "a"

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logging_file,
        filemode=filemode,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    logger = partial(logger_fcn, logger=logger)

    #### Autosave ####

    autosave_reciever, autosave_sender = Pipe(duplex=False)
    autosave_prc = Process(target=autosave, args=(autosave_seconds, autosave_sender))

    #### Training variables intitalization ####

    create_test_files = not (exists(test_data_path))

    lowest_test_loss = float("inf")
    last_test_loss = float("inf")
    best_epoch = 0
    progressive_epoch = 0
    patience = 5
    log_interval = int(np.round(no_train_batches / 10))

    data_namespace = Namespace()
    data_namespace.train_data_path = train_data_path
    data_namespace.test_data_path = test_data_path
    data_namespace.no_data_train = no_data_train
    data_namespace.no_data_test = no_data_test
    data_namespace.train_shuffle_seeds = train_shuffle_seeds
    data_namespace.train_indices = train_indices
    data_namespace.test_indices = test_indices
    data_namespace.train_batches_indices = train_batches_indices
    data_namespace.test_batches_indices = test_batches_indices
    data_namespace.no_train_batches = no_train_batches
    data_namespace.no_test_batches = no_test_batches
    data_namespace.batch_size = batch_size
    data_namespace.train_data_path = train_data_path

    preprocess_namespace = Namespace()
    preprocess_namespace.melSpec = melSpec
    preprocess_namespace.feat = feat
    preprocess_namespace.kmeans = kmeans
    preprocess_namespace.window_width = window_width

    if checkpointfile:
        model_state = ModelState.load(checkpointfile + ".npz")
        batch_i_offset = model_state.batch_i
        state_dict_file = model_state.previousfile
    else:
        model_state = ModelState()
        model_state.model_state_dict = model.state_dict()
        model_state.lr = lr
        model_state.best_epoch = best_epoch
        model_state.lowest_test_loss = lowest_test_loss
        model_state.last_test_loss = last_test_loss
        model_state.progressive_epoch = progressive_epoch
        model_state.patience = patience
        model_state.epoch_i = 0
        model_state.batch_i = 0
        batch_i_offset = None
        state_dict_file = None

    model_namespace = Namespace()
    model_namespace.model = model
    if state_dict_file:
        model_namespace.model.load_state_dict(torch.load(state_dict_file + ".pth"))
    model_namespace.lossfn = lossfn
    model_namespace.optimizer = optimizer
    model_namespace.lr = model_state.lr
    model_namespace.best_epoch = model_state.best_epoch
    model_namespace.lowest_test_loss = model_state.lowest_test_loss
    model_namespace.last_test_loss = model_state.last_test_loss
    model_namespace.progressive_epoch = model_state.progressive_epoch
    model_namespace.patience = model_state.patience
    model_namespace.log_interval = log_interval
    model_namespace.num_classes = num_classes
    model_namespace.init_class = init_class
    model_namespace.device = device
    model_namespace.epoch_i = model_state.epoch_i
    model_namespace.batch_i = model_state.batch_i

    train_prc = Process(
        target=train_data_gen_proc,
        args=(
            data_namespace,
            preprocess_namespace,
            epochs,
            model_state.epoch_i,
            model_state.batch_i,
        ),
    )

    #### Parallel Processes Start ####
    train_prc.start()
    autosave_prc.start()

    #### Training ####

    with open("tqdm_epoch.log", "w") as file:
        progress_bar = tqdm(
            range(model_state.epoch_i, epochs),
            file=file,
            bar_format="[{elapsed}<{remaining}, " "{rate_fmt}{postfix}]",
        )

        for epoch_i in progress_bar:
            model_namespace.epoch_i = epoch_i

            epoch_str = f"Epoch: {model_namespace.epoch_i:04d}"
            lr_str = f"Lr: {model_namespace.lr:.4g}"
            logger(f"{epoch_str} {lr_str} {progress_bar}", right="=")

            train_model(
                model_namespace,
                data_namespace,
                model_state,
                batch_i_offset,
                autosave_reciever,
                logger,
            )
            batch_i_offset = None

            test_loss, test_loss = evaluate_model(
                model_namespace,
                data_namespace,
                preprocess_namespace,
                create_test_files,
                logger,
            )

            #### Early Stopping and Saving ####

            if model_namespace.last_test_loss * 0.99 >= test_loss:
                logger(f"Progressive Epoch!")
                model_namespace.progressive_epoch = epoch_i
            else:
                logger(f"Not a Progressive Epoch!")
            model_namespace.last_test_loss = test_loss

            if test_loss < lowest_test_loss:
                model_namespace.best_epoch = epoch_i
                model_namespace.lowest_test_loss = test_loss
                torch.save(model_namespace.model.state_dict(), "model.pth")
                logger("*")
                logger(f"Best Epoch! Model Saved", right="*")
                logger("*")
            if (
                model_namespace.epoch_i - model_namespace.progressive_epoch
                >= model_namespace.patience
            ):
                logger("&")
                logger(
                    f"Early stopping after {patience} epochs of no significant improvement"
                )
                logger("&")
                break

            #### Learning Rate Scheduler ####

            scheduler.step()
            lr1 = scheduler.get_last_lr()[0]
            if lr1 != lr:
                model_namespace.lr = lr1
                logger(f"New learning rate: {model_namespace.lr:.4g}", right=" ")
            logger("=")
