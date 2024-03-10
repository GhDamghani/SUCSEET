from os.path import join
import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    num_classes = 20
    feature_folder = "../Dataset_Word"  # "../Kmeans of sounds"
    path_input = feature_folder  # join(feature_folder, "features")
    participant = "sub-06"  #  "p07_ses1_sentences"
    feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
    cluster = np.load(join(path_input, f"{participant}_spec_cluster_{num_classes}.npy"))

    X_train, X_test, y_train, y_test = train_test_split(
        feat, cluster, test_size=0.2, random_state=0
    )

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
