import numpy as np
from os.path import join


def rms(x):
    return np.sqrt(np.mean(x**2))


if __name__ == "__main__":
    num_classes = 20
    feature_folder = "../Dataset_Word"  # "../Kmeans of sounds"
    path_input = feature_folder  # join(feature_folder, "features")
    participant = "sub-06"  #  "p07_ses1_sentences"
    feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))

    feat_rms = np.apply_along_axis(rms, 1, feat)

    speech_mask = cluster != 0
    feat_rms_speech = feat_rms[speech_mask]
    feat_rms_silent = feat_rms[~speech_mask]

    threshold = (np.mean(feat_rms_speech) + np.mean(feat_rms_silent)) / 2

    y_pred_speech = feat_rms > threshold
    y_pred = np.zeros(cluster.shape)
    y_pred[y_pred_speech] = 1
    cluster[speech_mask] = 1
    acc = np.sum(y_pred == cluster) / len(y_pred)
    print(threshold)
    print(acc)
