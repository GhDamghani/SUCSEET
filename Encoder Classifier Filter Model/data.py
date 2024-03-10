import torch
import numpy as np
from prefetch_generator import background


def gen_batches(n, batch_size, *, min_batch_size=0):
    "from sklearn.utils"
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


class MyIterableDataset:

    def __init__(
        self,
        indices,
        feat,
        cluster,
        timepoints,
        num_eeg_channels,
        batch_size,
        epochs,
        device,
        offset=0,
    ):

        self.indices = indices
        self.feat = feat
        self.cluster = cluster
        self.timepoints = timepoints
        self.num_eeg_channels = num_eeg_channels
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_i = 0
        self.device = device
        self.offset = offset

    def preprocess(self, i):
        x1 = self.feat[i : i + self.timepoints]
        y1 = self.cluster[i + self.timepoints - 1 : i + self.timepoints] - 1
        return x1, y1

    @background(max_prefetch=3)
    def __iter__(self):
        np.random.shuffle(self.indices)
        iterator = enumerate(gen_batches(len(self.indices), self.batch_size))
        for batch_i, batch in iterator:
            if batch_i < self.offset:
                continue
            ind = self.indices[batch]
            X, y = zip(*(self.preprocess(i) for i in ind))
            X = torch.tensor(np.stack(X)).to(self.device)
            y = torch.tensor(np.stack(y), dtype=torch.long).squeeze(
                -1
            )  # .to(self.device)
            yield X, y
        self.offset = 0

    def __len__(self):
        return len(self.indices)


def remove_silences(indices: list, removes: set) -> None:
    indices[:] = [x for x in indices if x not in removes]


def get_train_val_datasets(
    feat,
    cluster,
    timepoints,
    num_eeg_channels,
    batch_size,
    epochs,
    device,
    validation_ratio=0.2,
    p_sample=1.0,
):

    no_samples = len(feat)

    silences = np.array([i for i, x in enumerate(cluster) if (x == 0)])  #  or x == 1
    remove_indices = silences - timepoints + 1
    remove_indices = remove_indices[remove_indices >= 0]

    val_clip_length = np.int32(np.round(no_samples * validation_ratio))
    assert val_clip_length > timepoints
    val_start_range = range(timepoints, no_samples - timepoints - val_clip_length + 1)
    val_start = np.random.choice(val_start_range, 1).item()
    val_indices = list(range(val_start, val_start + val_clip_length - timepoints + 1))

    train_indices_left = list(range(0, val_start - timepoints + 1))
    train_indices_right = list(
        range(val_start + val_clip_length, no_samples - timepoints + 1)
    )
    train_indices = train_indices_left + train_indices_right

    remove_silences(val_indices, remove_indices)
    remove_silences(train_indices, remove_indices)

    if 0 < p_sample < 1:
        # Only a portion of train data to test the general capability of the model at first, optional
        train_indices = np.random.choice(
            train_indices, int(len(train_indices) * p_sample), replace=False
        )
        val_indices = np.random.choice(
            val_indices, int(len(val_indices) * p_sample), replace=False
        )

    return MyIterableDataset(
        train_indices,
        feat,
        cluster,
        timepoints,
        num_eeg_channels,
        batch_size,
        epochs,
        device,
    ), MyIterableDataset(
        val_indices,
        feat,
        cluster,
        timepoints,
        num_eeg_channels,
        batch_size,
        epochs,
        device,
    )
