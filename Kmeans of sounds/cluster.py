from sklearn.cluster import KMeans
import numpy as np
from os.path import join
import joblib
import matplotlib.pyplot as plt

# from slider import slider

path_input = r"./features"
participant = "sub-06"

melSpec = np.load(join(path_input, f"{participant}_spec.npy"))

# trainData = slider(melSpec)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
dists = kmeans.fit_transform(melSpec)
lbl = kmeans.labels_
# keep=[]
# for i in range(dists.shape[1]):
#     keep.append(np.argmin(dists[:,i]))
print(dists.shape[1])

lbl_hist = np.unique(lbl, return_counts=True)
plt.figure()
plt.stem(lbl_hist[0], lbl_hist[1] / np.sum(lbl_hist[1]))
plt.xlabel("Labels")
plt.ylabel("Freq")
plt.title("Histogram of True Output for the whole data")

plt.tight_layout()

joblib.dump(kmeans, "kmeans.joblib")
plt.show()
