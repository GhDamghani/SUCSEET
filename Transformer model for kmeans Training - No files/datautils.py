from os import listdir, mkdir, remove
import numpy as np
from os.path import join, exists
import torch
import torch.nn.functional as F
from multiprocessing import Process


def train_test_label(
    no_samples, window_width, test_all_ratio=0.25, p_sample=1.0, seed=1534
):
    l_test = np.int32(np.round(no_samples * test_all_ratio))
    assert l_test > window_width
    test_pick_list = list(
        range(window_width, no_samples - window_width - l_test + 1)
    )  # + [0, -1]
    np.random.seed(seed)  # reproducibility
    test_index_clip = np.random.choice(test_pick_list, 1).item()
    train_indices = list(range(0, test_index_clip - window_width)) + list(
        range(test_index_clip + l_test, no_samples - window_width + 1)
    )
    test_indices = list(
        range(test_index_clip, test_index_clip + l_test - window_width + 1)
    )

    if 0 < p_sample < 1:
        # Only a portion of train data to test the general capability of the model at first, optional
        np.random.seed(28)  # reproducibility
        train_indices = np.random.choice(
            train_indices, int(len(train_indices) * p_sample), replace=False
        )
        test_indices = np.random.choice(
            test_indices, int(len(test_indices) * p_sample), replace=False
        )

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return train_indices, test_indices


def read_and_preprocess(preprocess_namespace, s_ind):
    x1 = preprocess_namespace.feat[s_ind : s_ind + preprocess_namespace.window_width]
    y0 = preprocess_namespace.melSpec[s_ind : s_ind + preprocess_namespace.window_width]
    y1 = preprocess_namespace.kmeans.predict(y0)
    return x1, y1


def batch_tensor(preprocess_namespace, device, trainname_batch):
    x_batch = []
    y_batch = []
    for s_ind in trainname_batch:  # preprocessing
        x1, y1 = read_and_preprocess(preprocess_namespace, s_ind)
        x_batch.append(x1)
        y_batch.append(y1)

    X = torch.tensor(np.array(x_batch).astype("float32")).to(device)
    y = torch.tensor(np.array(y_batch).flatten().astype("int64")).to(device)
    return X, y


def train_data_gen(
    data_namespace, preprocess_namespace, device, epoch_i, batch_i_offset
) -> None:  # melSpec, feat, kmeans, w
    random_indices = np.arange(data_namespace.no_data_train, dtype=int)
    np.random.seed(data_namespace.train_shuffle_seeds[epoch_i])
    np.random.shuffle(random_indices)
    dataname_train_shuffled = data_namespace.train_indices.copy()[random_indices]
    for batch_i, x in enumerate(data_namespace.train_batches_indices):
        if batch_i_offset:
            if batch_i < batch_i_offset:
                continue
            batch_i_offset = None
        trainname_batch = dataname_train_shuffled[x]
        yield batch_tensor(preprocess_namespace, device, trainname_batch)


def test_data_gen(data_namespace, preprocess_namespace, device):
    for i, x in enumerate(data_namespace.test_batches_indices):
        trainname_batch = data_namespace.test_indices[x]
        x_batch = []
        y_batch = []
        for s_ind in trainname_batch:  # preprocessing
            x1, y1 = read_and_preprocess(preprocess_namespace, s_ind)
            x_batch.append(x1)
            y_batch.append(y1)

        X, y = batch_tensor(preprocess_namespace, device, trainname_batch)
        yield batch_tensor(preprocess_namespace, device, trainname_batch)
