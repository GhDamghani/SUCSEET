import numpy as np
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler


def normalize(train, whole):
    normalizer = StandardScaler().fit(train)
    whole = normalizer.transform(whole)
    return whole


def DCT(train, whole):
    mfccs = dct(whole, type=2, axis=1, norm="ortho")
    return mfccs
