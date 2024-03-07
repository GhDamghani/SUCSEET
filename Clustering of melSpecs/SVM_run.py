from os.path import join
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":
    kmeans_folder = ""
    path_input = "features"
    participant = "sub-06"
    melSpec = np.load(join(path_input, f"{participant}_spec.npy"))
    feat = np.load(join(path_input, f"{participant}_feat.npy"))
    kmeans = joblib.load(join(kmeans_folder, "kmeans.joblib"))

    labels = kmeans.predict(melSpec)

    X_train, X_test, y_train, y_test = train_test_split(
        feat, labels, test_size=0.2, random_state=42
    )

    clf = make_pipeline(
        StandardScaler(), LinearSVC(dual="auto", random_state=0, tol=1e-5)
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
