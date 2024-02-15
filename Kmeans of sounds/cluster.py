from sklearn.cluster import KMeans
import numpy as np
from os.path import join
import joblib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from reconstruction_minimal import createAudio

# from slider import slider

path_input = r"./features"
participant = "sub-06"
audiosr = 16000

melSpec = np.load(join(path_input, f"{participant}_spec.npy"))

# trainData = slider(melSpec)

kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto")
dists = kmeans.fit_transform(melSpec)
lbl = kmeans.labels_
# keep=[]
# for i in range(dists.shape[1]):
#     keep.append(np.argmin(dists[:,i]))
print(dists.shape[1])

center = lambda x: kmeans.cluster_centers_[x]
center_melSpec = np.stack(tuple(center(x) for x in lbl), axis=0)
center_audio = createAudio(center_melSpec, audiosr)
origWav = createAudio(melSpec, audiosr=audiosr)


np.save(join(path_input, f"{participant}_melSpec_cluster.npy"), lbl)
np.save(
    join(path_input, f"{participant}_melSpec_cluster_centers.npy"),
    kmeans.cluster_centers_,
)
wavfile.write(
    join(path_input, f"{participant}_orig_synthesized.wav"), int(audiosr), origWav
)
wavfile.write(
    join(path_input, f"{participant}_cluster_center_reconstructed.wav"),
    int(audiosr),
    center_audio,
)

plt.figure()
ax1 = plt.subplot(2, 1, 1)
plt.plot(lbl[:2000])
plt.subplot(2, 1, 2, sharex=ax1)
plt.tight_layout()
plt.imshow(melSpec[:2000].T, "gray", aspect="auto")

lbl_hist = np.unique(lbl, return_counts=True)
hist = lbl_hist[1]
np.save("histogram.npy", hist)
hist = lbl_hist[1] / np.sum(lbl_hist[1])
plt.figure()
plt.stem(
    lbl_hist[0],
    lbl_hist[1] / np.sum(lbl_hist[1]),
)
plt.xlabel("Labels")
plt.ylabel("Freq")
plt.title("Histogram of True Output for the whole data")

plt.tight_layout()
plt.savefig("histogram.png")

joblib.dump(kmeans, "kmeans.joblib")
plt.show()
