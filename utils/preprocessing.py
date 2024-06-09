import numpy as np
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKLearnPCA


def normalize(train, whole):
    normalizer = StandardScaler().fit(train)
    whole = normalizer.transform(whole)
    return whole


def PCA(train, whole):
    pca = SKLearnPCA()
    pca.fit(train)
    whole = pca.transform(whole)
    return whole


def DCT(_, X):
    mfccs = dct(X, type=2, axis=1, norm="ortho")
    return mfccs
