from sklearn.cluster import KMeans
import numpy as np
from os.path import join
import joblib
from slider import slider

path_input = r'./features'
participant = 'sub-06'

melSpec = np.load(join(path_input, f'{participant}_spec.npy'))

trainData = slider(melSpec)

kmeans = KMeans(n_clusters=50, random_state=0, n_init="auto")
dists = kmeans.fit_transform(trainData)
lbl = kmeans.labels_
keep=[]
for i in range(dists.shape[1]):
    keep.append(np.argmin(dists[:,i]))

joblib.dump(kmeans, 'kmeans.joblib')