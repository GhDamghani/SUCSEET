import numpy as np
from os.path import join

feat_path = r"./features"
pts = ["sub-%02d" % i for i in range(1, 11)]
for pNr, pt in enumerate(pts):
    # Load the data
    data = np.load(join(feat_path, f"{pt}_feat.npy"))
    print(pNr, data.shape)
