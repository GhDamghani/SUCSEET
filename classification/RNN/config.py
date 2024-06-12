from functools import partial
from torch import optim, device, cuda
import scipy
import numpy as np
from os.path import join, exists
from os import makedirs
from functools import partial
import loss_metrics

from sklearn.model_selection import KFold

import sys

master_path = "../.."
sys.path.append(master_path)
import utils


class Config:
    def __init__(self, **kwargs) -> None:
        self.model_task = "classification"
        self.nfolds = 10
        self.dataset_name = "Word"
        self.vocoder_name = "VocGAN"  # "Griffin_Lim"
        self.participant = "sub-06"  # "p07_ses1_sentences"

        self.SEED = 7
        self.DEVICE = device("cuda" if cuda.is_available() else "cpu")

        self.clustering_method = "kmeans"
        self.clf_name = "RNN_test"

        self.num_classes = 2
        self.window_size = 40
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.1
        self.optimizer = partial(optim.Adam, lr=1e-5, weight_decay=1e-2, amsgrad=True)
        self.BATCH_SIZE = 32
        self.EPOCHS = 1

        self.output_indices = [-1]

        self.LOG_TIMES = 10

        self.topk = 3

    def update(self):
        self.input_path = join(
            master_path, "dataset", self.dataset_name, self.vocoder_name
        )
        output_path = join(
            master_path,
            "results",
            self.model_task,
            f"{self.window_size}",
            self.clf_name,
        )

        self.file_names = utils.names.Names(
            output_path,
            self.dataset_name,
            self.vocoder_name,
            self.participant,
            self.nfolds,
            self.num_classes,
            self.clustering_method,
        )
        self.file_names.update(fold=self.fold)

    def proc(self):
        self.update()
        self.feat = np.load(join(self.input_path, f"{self.participant}_feat.npy"))
        self.metrics = {"total": loss_metrics.total}

        self.feat_size = self.feat.shape[1]
        with np.load(
            join(
                master_path,
                "results",
                "clustering",
                self.clustering_method,
                f"{self.dataset_name}_{self.vocoder_name}_{self.participant}_spec_c{self.num_classes:02d}_f{self.nfolds:02d}.npz",
            )
        ) as data:

            cluster_train = data[f"y_train_{self.fold:02d}"]
            cluster_test = data[f"y_test_{self.fold:02d}"]
            self.melSpec_centers = data[f"centers_{self.fold:02d}"]
        for fold, (train_index, test_index) in enumerate(
            KFold(n_splits=self.nfolds, shuffle=False).split(self.feat)
        ):
            if fold != self.fold:
                continue
            self.cluster = np.empty((cluster_train.shape[0] + cluster_test.shape[0],))
            self.cluster[train_index] = cluster_train
            self.cluster[test_index] = cluster_test

        self.melSpec = np.load(join(self.input_path, f"{self.participant}_spec.npy"))
        self.metrics["corrects"] = loss_metrics.corrects

        makedirs(self.file_names.output_path, exist_ok=True)
        makedirs("logs", exist_ok=True)
