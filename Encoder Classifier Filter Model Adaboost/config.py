from functools import partial
from torch import optim, device, cuda

import numpy as np
from os.path import join, exists
import model_loss

num_classes = 20

optimizer = partial(optim.Adam, lr=1e-5, weight_decay=1e-1, amsgrad=True)
lr_scheduler = partial(optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, patience=100)

feature_folder = "../Dataset_Word"  # "../Kmeans of sounds"
path_input = feature_folder  # join(feature_folder, "features")
kmeans_folder = feature_folder
participant = "sub-06"  #  "p07_ses1_sentences"
feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))
cluster_centers = np.load(
    join(path_input, f"{participant}_spec_cluster_{num_classes}_centers.npy")
)
histogram_weights = np.load(
    join(path_input, f"{participant}_spec_cluster_{num_classes}_hist.npy")
)[1:]

criterion = model_loss.criterion_cross_entropy_observation_weight_class_balanced(
    histogram_weights, num_classes - 1
)
# criterion = model_loss.CCE(balancing_factor=1 / (num_classes - 1))

# criterion = model_loss.criterion(
#     histogram_weights, num_classes, weights=True
# )  # num_classes

SEED = 1379456
DEVICE = device("cuda" if cuda.is_available() else "cpu")

d_model = 128
dim_feedforward = 2048
num_heads = 4
num_layers = 20
timepoints = 5
num_classes = num_classes
num_eeg_channels = feat.shape[1]

dropout_prenet = 0.1
dropout_encoder = 0.1
dropout_clf = 0.1

VALIDATION_RATIO = 0.2
P_SAMPLE = 1.0

BATCH_SIZE = 64
EPOCHS = 2000
LOG_TIMES = 10

AUTOSAVE_SECONDS = 600
CHECKPOINTFILE = None
EARLY_STOPPING_PATIENCE = 20
