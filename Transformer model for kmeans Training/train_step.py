from tqdm import tqdm
from os.path import join
import torch
from os import remove
from multiprocessing import Process
from datautils import test_data_gen, get_data
import numpy as np


def train_model(
    model_namespace,
    data_namespace,
    model_state,
    batch_i_offset,
    autosave_reciever,
    logger,
):
    model_namespace.model.train()
    train_loss = 0
    corrects = 0
    total = 0

    if batch_i_offset:
        iterator = range(batch_i_offset, data_namespace.no_train_batches)
        batch_i_offset = None
    else:
        iterator = range(data_namespace.no_train_batches)

    with open("tqdm_batch.log", "w") as file:
        progress_bar = tqdm(
            iterator,
            file=file,
            bar_format="[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        for batch_i in progress_bar:
            model_namespace.batch_i = batch_i
            if autosave_reciever.poll():
                if autosave_reciever.recv() == True:
                    model_state.make_checkpoint(model_namespace)
                    model_state.save(model_namespace.model)

            filename = join(
                data_namespace.train_data_path,
                f"train_E{model_namespace.epoch_i:04}_B{batch_i:04}.npz",
            )
            X, y = get_data(
                filename, model_namespace.device, model_namespace.num_classes
            )

            # Compute prediction error
            pred = model_namespace.model(X, model_namespace.init_class)
            pred = pred.reshape(-1, model_namespace.num_classes)
            loss = model_namespace.lossfn(pred, y)

            # Backpropagation
            model_namespace.optimizer.zero_grad()
            loss.backward()
            model_namespace.optimizer.step()

            train_loss += loss.item()
            pred_label = torch.argmax(pred, -1)
            corrects += torch.sum(y == pred_label).item()
            total += y.numel()

            try:
                remove(filename)
            except FileNotFoundError:
                pass

            if (batch_i + 1) % model_namespace.log_interval == 0:
                accuracy = corrects / total
                loss_str = f"Loss: {train_loss/total:.5g}"
                acc_str = f"Accuracy: {accuracy:7.2%}"
                left_str = f"{loss_str:12} | {acc_str:12}"
                right_str = str(progress_bar)
                logger(left_str, right=right_str)

    accuracy = corrects / total
    logger(
        f"Epoch {model_namespace.epoch_i} Train loss: {train_loss/total:.5g} Accuracy: {accuracy:02.2%}"
    )

    return


def evaluate_model(
    model_namespace,
    data_namespace,
    preprocess_namespace,
    create_test_files,
    logger,
    return_data=False,
):
    model_namespace.model.eval()

    if create_test_files:
        test_prc = Process(
            target=test_data_gen,
            args=(data_namespace, preprocess_namespace),
        )
        test_prc.start()

    test_loss = 0
    corrects = 0
    total = 0
    if return_data:
        pred_labels = []
        y_labels = []
    with torch.no_grad():
        for batch_i in range(data_namespace.no_test_batches):
            filename = join(data_namespace.test_data_path, f"test_B{batch_i:04}.npz")

            X, y = get_data(
                filename, model_namespace.device, model_namespace.num_classes
            )

            # Compute prediction error
            pred = model_namespace.model(X).reshape(-1, model_namespace.num_classes)
            test_loss += model_namespace.lossfn(pred, y).item()

            pred_label = torch.argmax(pred, -1)
            corrects += torch.sum(y == pred_label).item()
            total += y.numel()
            if return_data:
                pred_labels.append(pred_label.to("cpu").numpy())
                y_labels.append(y.to("cpu").numpy())

    test_acc = corrects / total
    test_loss /= total
    if return_data:
        logger(
            f"Evaluation Complete. Test loss: {test_loss:.5g} Accuracy: {test_acc:02.2%}"
        )
    else:
        logger(
            f"Epoch {model_namespace.epoch_i}  Test loss: {test_loss:.5g} Accuracy: {test_acc:02.2%}"
        )
    if return_data:
        pred_labels = np.concatenate(pred_labels)
        y_labels = np.concatenate(y_labels)
        return test_loss, test_acc, pred_labels, y_labels

    return test_loss, test_acc
