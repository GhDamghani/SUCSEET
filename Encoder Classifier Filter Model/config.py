from functools import partial
from torch import optim, device, cuda

import numpy as np
from os.path import join

from model_loss import CCE as criterion

num_classes = 20

optimizer = partial(optim.Adam, lr=1e-5, weight_decay=1e-2, amsgrad=True)
lr_scheduler = partial(optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, patience=100)

feature_folder = r"..\Dataset_Word"  # "../Kmeans of sounds"
path_input = feature_folder  # join(feature_folder, "features")
kmeans_folder = feature_folder
participant = "sub-06"  # "p07_ses1_sentences"
feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))
cluster_centers = np.load(
    join(path_input, f"{participant}_spec_cluster_{num_classes}_centers.npy")
)
histogram_weights = np.load(
    join(path_input, f"{participant}_spec_cluster_{num_classes}_hist.npy")
)

SEED = 1379456
DEVICE = device("cuda" if cuda.is_available() else "cpu")

d_model = 128
dim_feedforward = d_model * 8
num_heads = 4
num_layers = 5
timepoints = 400
num_classes = num_classes
num_eeg_channels = 127

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
