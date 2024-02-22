from functools import partial
from torch import optim, device, cuda

import numpy as np
from os.path import join

from model_loss import CCE as criterion


optimizer = partial(optim.Adam, lr=1e-5, weight_decay=1e-2, amsgrad=True)
lr_scheduler = partial(optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, patience=100)

feature_folder = r"..\sentences_data\eeg"  # "../Kmeans of sounds"
path_input = feature_folder  # join(feature_folder, "features")
kmeans_folder = r"..\Kmeans of sounds"
participant = "p07_ses1_sentences"  # "sub-06"
feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
cluster = np.load(join(path_input, f"{participant}_melSpec_cluster.npy"))
cluster_centers = np.load(
    join(path_input, f"{participant}_melSpec_cluster_centers.npy")
)

SEED = 1379456
DEVICE = device("cuda" if cuda.is_available() else "cpu")

d_model = 128
dim_feedforward = d_model * 8
num_heads = 4
num_layers = 5
timepoints = 400
num_classes = 20
num_eeg_channels = 130

dropout_prenet = 0.1
dropout_encoder = 0.1
dropout_clf = 0.1

VALIDATION_RATIO = 0.2
P_SAMPLE = 1.0

BATCH_SIZE = 64
EPOCHS = 200
LOG_TIMES = 10

AUTOSAVE_SECONDS = 600
CHECKPOINTFILE = None
EARLY_STOPPING_PATIENCE = 100
