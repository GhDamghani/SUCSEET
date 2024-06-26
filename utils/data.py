from sklearn.model_selection import KFold
from prefetch_generator import background
import numpy as np
from . import preprocessing


class WindowedDataset:
    def __init__(
        self,
        X,
        y,
        window_size,
        output_indices,
    ) -> None:
        self.X = X
        self.y = y
        self.window_size = window_size
        self.output_indices = output_indices
        self.output_size = len(output_indices)
        self.arg = np.arange(len(X))

    def shuffle(self):
        """Shuffle both X and y but keeping the correspondence between the indices"""
        indices = np.random.permutation(len(self.X))
        self.arg = self.arg[indices]

    def sort(self):
        argsort = np.argsort(self.arg)
        self.arg = self.arg[argsort]

    def __getitem__(self, index):
        return (
            self.X[self.arg[index] : self.arg[index] + self.window_size],
            self.y[self.arg[index] : self.arg[index] + self.window_size][
                self.output_indices
            ],
        )

    def __len__(self):
        return len(self.X) + 1 - self.window_size

    @background(max_prefetch=10)
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def generate_batch(self, batch_size, shuffle=False, sort=False):
        """
        Generates a batch of data for training or evaluation.

        Parameters:
            batch_size (int): The number of samples in each batch. If set to -1, the entire dataset is used.
            shuffle (bool, optional): Whether to shuffle the data before generating the batches. Defaults to False.
            sort (bool, optional): Whether to sort the data before generating the batches. Defaults to False.

        Yields:
            X (ndarray): A batch of input data with shape (batch_size, window_size, X.shape[1]).
            y (ndarray): A batch of target data with shape (batch_size, y.shape[1]).

        """

        if batch_size == -1:
            batch_size = len(self)
        if shuffle:
            self.shuffle()
        if sort:
            self.sort()
        X = np.empty(
            (batch_size, self.window_size, self.X.shape[1]), dtype=self.X.dtype
        )
        y = np.empty(
            (batch_size, self.output_size, *self.y.shape[1:]), dtype=self.y.dtype
        )
        for i, (x0, y0) in enumerate(self):
            X[i % batch_size] = x0
            y[i % batch_size] = y0
            if (i + 1) % batch_size == 0:
                yield X, y
        if i % batch_size > 0:
            yield X[: (i + 1) % batch_size], y[: (i + 1) % batch_size]


class WindowedMultiDataset:
    def __init__(self, *datasets) -> None:
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.lengths_accum = np.cumsum(self.lengths)
        self.arg = np.arange(self.lengths_accum[-1])

    def __len__(self):
        return self.lengths_accum[-1]

    def __getitem__(self, index):
        i = np.where(self.lengths_accum > self.arg[index])[0][0]
        return self.datasets[i][
            self.arg[index] - self.lengths[i - 1] if i > 0 else self.arg[index]
        ]

    def shuffle(self):
        indices = np.random.permutation(len(self))
        self.arg = self.arg[indices]

    def sort(self):
        argsort = np.argsort(self.arg)
        self.arg = self.arg[argsort]

    def generate_batch(self, batch_size, shuffle=False, sort=False):
        """
        Generates a batch of data for training or evaluation.

        Parameters:
            batch_size (int): The size of the batch. If set to -1, the entire dataset is used.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            sort (bool, optional): Whether to sort the data. Defaults to False.

        Yields:
            tuple: A tuple containing the input data (X) and the corresponding target data (y) for each batch.
                - X (ndarray): The input data of shape (batch_size, window_size, feature_size).
                - y (ndarray): The target data of shape (batch_size, target_size).

        """

        if batch_size == -1:
            batch_size = len(self)
        if shuffle:
            self.shuffle()
        if sort:
            self.sort()
        X = np.empty(
            (batch_size, self.datasets[0].window_size, self.datasets[0].X.shape[1]),
            dtype=self.datasets[0].X.dtype,
        )
        y = np.empty(
            (batch_size, self.datasets[0].output_size, *self.datasets[0].y.shape[1:]),
            dtype=self.datasets[0].y.dtype,
        )
        for i, x in enumerate(self.arg):
            x0, y0 = self[x]
            X[i % batch_size] = x0
            y[i % batch_size] = y0
            if (i + 1) % batch_size == 0:
                yield X, y
        if i % batch_size > 0:
            yield X[: i % batch_size], y[: i % batch_size]


class WindowedData:

    def __init__(
        self,
        X,
        y,
        window_size=1,
        num_folds=10,
        output_indices=(-1,),
    ) -> None:
        self.X = X
        if y.shape[0] != num_folds:
            y = np.stack([y] * num_folds, axis=0)
        self.y = y
        self.window_size = window_size
        self.kf = KFold(n_splits=num_folds, shuffle=False)
        self.output_indices = output_indices

    @staticmethod
    def segment(ind):
        segments = []
        start = 0
        for i in range(1, len(ind)):
            if ind[i] != ind[i - 1] + 1:
                segments.append(ind[start:i])
                start = i
        segments.append(ind[start:])
        return segments

    def __iter__(self):
        for fold, (train_index, test_index) in enumerate(self.kf.split(self.X)):
            X = self.X
            test_dataset = WindowedDataset(
                X[test_index],
                self.y[fold, test_index],
                self.window_size,
                self.output_indices,
            )
            train_segments = self.segment(train_index)
            train_datasets = []
            for train_segment in train_segments:
                train_dataset = WindowedDataset(
                    X[train_segment],
                    self.y[fold, train_segment],
                    self.window_size,
                    self.output_indices,
                )
                train_datasets.append(train_dataset)
            train_dataset = WindowedMultiDataset(*train_datasets)
            yield train_dataset, test_dataset  # , (train_index, test_index)
