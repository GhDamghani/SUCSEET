import torch
import numpy as np
from prefetch_generator import background
from sklearn.utils import gen_batches


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
        shuffle_seed=False,
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
        self.shuffle_seed = shuffle_seed
        self.device = device
        self.offset = offset
        if self.shuffle_seed:
            np.random.seed(shuffle_seed)
            self.shuffle_indices = np.random.randint(1, 100000, epochs)

    def preprocess(self, i, disturb=False):
        x1 = self.feat[i : i + self.timepoints]
        # x1[:, sw[0]], x1[:, sw[1]] = x1[:, sw[1]], x1[:, sw[0]]
        # if disturb:
        #     x1 = x1 * np.random.uniform(0.8, 1.2, (1, self.num_eeg_channels)).astype(
        #         np.float32
        #     ) + np.random.normal(
        #         0, 0.8, (self.timepoints, self.num_eeg_channels)
        #     ).astype(
        #         np.float32
        #     )

        y1 = self.cluster[i + self.timepoints - 1 : i + self.timepoints]
        return x1, y1

    @background(max_prefetch=3)
    def __iter__(self):
        disturb = False
        if self.shuffle_seed:
            np.random.seed(self.shuffle_indices[self.epoch_i])
            np.random.shuffle(self.indices)
            disturb = False
        iterator = enumerate(gen_batches(len(self.indices), self.batch_size))
        for batch_i, batch in iterator:
            if batch_i < self.offset:
                continue
            ind = self.indices[batch]
            X, y = zip(*(self.preprocess(i, disturb) for i in ind))
            X = torch.tensor(np.stack(X)).to(self.device)
            y = torch.tensor(np.stack(y), dtype=torch.long)  # .to(self.device)
            yield X, y
        self.offset = 0

    # Assuming 'generator' is your input generato

    def __len__(self):
        return len(self.indices)


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

    l_val = np.int32(np.round(no_samples * validation_ratio))
    assert l_val > timepoints
    val_pick_list = list(
        range(timepoints, no_samples - timepoints - l_val + 1)
    )  # + [0, -1]
    val_index_clip = np.random.choice(val_pick_list, 1).item()
    train_indices = list(range(0, val_index_clip - timepoints)) + list(
        range(val_index_clip + l_val, no_samples - timepoints + 1)
    )
    val_indices = list(range(val_index_clip, val_index_clip + l_val - timepoints + 1))

    if 0 < p_sample < 1:
        # Only a portion of train data to test the general capability of the model at first, optional
        train_indices = np.random.choice(
            train_indices, int(len(train_indices) * p_sample), replace=False
        )
        val_indices = np.random.choice(
            val_indices, int(len(val_indices) * p_sample), replace=False
        )

    # aug_swtiching = list(combinations(range(num_eeg_channels), 2))
    # random.seed(1379)
    # random.shuffle(aug_swtiching)
    # aug_swtiching = aug_swtiching[:100]

    # train_indices = list(product(train_indices, aug_swtiching))
    # val_indices = list(product(val_indices, aug_swtiching))

    return MyIterableDataset(
        train_indices,
        feat,
        cluster,
        timepoints,
        num_eeg_channels,
        batch_size,
        epochs,
        device,
        shuffle_seed=np.random.randint(1, 10000),
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
