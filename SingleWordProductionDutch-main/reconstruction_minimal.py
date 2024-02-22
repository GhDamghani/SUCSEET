import os

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import reconstructWave as rW
import MelFilterBank as mel


def createAudio(spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01):
    """
    Create a reconstructed audio wavefrom

    Parameters
    ----------
    spectrogram: array
        Spectrogram of the audio
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram was calculated
    frameshift: float
        Shift (in seconds) after which next window was extracted
    Returns
    ----------
    scaled: array
        Scaled audio waveform
    """
    mfb = mel.MelFilterBank(
        int((audiosr * winLength) / 2 + 1), spectrogram.shape[1], audiosr
    )
    nfolds = 10
    hop = int(spectrogram.shape[0] / nfolds)
    rec_audio = np.array([])
    for_reconstruction = mfb.fromLogMels(spectrogram)
    for w in range(0, spectrogram.shape[0], hop):
        spec = for_reconstruction[w : min(w + hop, for_reconstruction.shape[0]), :]
        rec = rW.reconstructWavFromSpectrogram(
            spec,
            spec.shape[0] * spec.shape[1],
            fftsize=int(audiosr * winLength),
            overlap=int(winLength / frameshift),
        )
        rec_audio = np.append(rec_audio, rec)
    scaled = np.int16(rec_audio / np.max(np.abs(rec_audio)) * 32767)
    return scaled


if __name__ == "__main__":
    feat_path = r"../Dataset_Word"
    result_path = r"../Dataset_Word"
    pts = ["sub-%02d" % i for i in range(1, 11)]

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000

    for pNr, pt in enumerate(pts):
        # Load the data
        spectrogram = np.load(os.path.join(feat_path, f"{pt}_spec.npy"))

        # Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)

        # For comparison synthesize the original spectrogram with Griffin-Lim
        origWav = createAudio(
            spectrogram, audiosr=audiosr, winLength=winLength, frameshift=frameshift
        )
        wavfile.write(
            os.path.join(result_path, f"{pt}_orig_synthesized.wav"),
            int(audiosr),
            origWav,
        )
