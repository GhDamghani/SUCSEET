import joblib
import numpy as np
import matplotlib.pyplot as plt
kmeans = joblib.load('kmeans.joblib')
lbl = kmeans.labels_
# Calculate the prior probabilities
unique_labels, counts = np.unique(lbl, return_counts=True)
total_points = len(lbl)
priors = counts / total_points

biggest_cluster = np.argmax(counts)
print(f'Biggest Cluster: {biggest_cluster}, prior: {priors[biggest_cluster]}')
print(f'Sum all probs: {np.sum(priors)}')

plt.stem(unique_labels, priors, linefmt='-', markerfmt='o', basefmt='k-')
plt.xticks(unique_labels)
plt.xlabel('Labels')
plt.ylabel('Prob')
plt.title('Histogram of Clusters')
plt.show()