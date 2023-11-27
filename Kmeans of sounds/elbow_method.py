from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from os.path import join, exists
from os import mkdir
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
# from slider import slider

path_input = r'./features'
participant = 'sub-06'
res_path = 'results'
if not exists(res_path): mkdir(res_path)

melSpec = np.load(join(path_input, f'{participant}_spec.npy'))

# trainData = slider(melSpec)

min_n_clusters = 5
max_n_clusters = 200
cluster_iter = range(min_n_clusters, max_n_clusters+1)

distortions = np.zeros((len(cluster_iter),))
sil = np.zeros((len(cluster_iter),))



for i, n_clu in tqdm(enumerate(cluster_iter)):
    kmeans = KMeans(n_clusters=n_clu, random_state=10, n_init="auto", max_iter=1000, verbose=0)
    dists = kmeans.fit_transform(melSpec)
    lbl = kmeans.labels_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(melSpec, kmeans.labels_)
    tqdm.write(f"{n_clu:03d} Clusters: Inertia = {inertia:10.2f}, Silhouette Score = {silhouette_avg:.4f}")
    distortions[i]= inertia
    sil[i]= silhouette_avg
    joblib.dump(kmeans, join(res_path, f'kmeans-{n_clu:03d}.joblib'))

plt.subplot(2, 1, 1)
plt.plot(cluster_iter, distortions)
plt.yscale('log')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

plt.subplot(2, 1, 2)
plt.plot(cluster_iter, sil)
plt.yscale('log')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

plt.show()

# joblib.dump(kmeans, 'kmeans.joblib')