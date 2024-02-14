from os.path import join
import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    kmeans_folder = ""
    path_input = "features"
    participant = "sub-06"
    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))
    kmeans = joblib.load(join(kmeans_folder, "kmeans.joblib"))

    labels = kmeans.predict(melSpec)

    X_train, X_test, y_train, y_test = train_test_split(
        feat, labels, test_size=0.25, random_state=42
    )

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
