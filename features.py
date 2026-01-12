import random
import os
from typing import Callable
import hashlib
import inspect
import traceback
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import librosa
import params
from IPython.display import clear_output
from tensorflow.keras import layers, models


def compute_mel_spectrogram(y: np.ndarray, n_mels=40):
    mel_spectrogram = librosa.feature.mfcc(y=y, sr=params.sample_rate, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db


def compute_spectral_centroid(y: np.ndarray):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=params.sample_rate)
    return spectral_centroid


def compute_spectral_bandwidth(y: np.ndarray):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=params.sample_rate)
    return spectral_bandwidth


def compute_spectral_rolloff(y: np.ndarray):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=params.sample_rate)
    return spectral_rolloff


def compute_spectral_contrast(y: np.ndarray):
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=params.sample_rate)
    return spectral_contrast


def compute_spectral_flatness(y: np.ndarray):
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    return spectral_flatness


def compute_tempo_features(y: np.ndarray) -> np.ndarray:
    oenv = librosa.onset.onset_strength(y=y)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv)
    tempi = librosa.tempo_frequencies(tempogram.shape[0])
    arr = tempi[np.argmax(tempogram[5:-5], axis=0)]
    arr[np.isinf(arr)] = 0
    return arr


def compute_chromagram(y: np.ndarray):
    return librosa.feature.chroma_stft(y=y)


def aggregate(y: np.ndarray) -> np.ndarray:
    aggregated_features = [
        y.min(axis=1),
        np.quantile(y, 0.25, axis=1),
        y.mean(axis=1),
        np.median(y, axis=1),
        np.quantile(y, 0.75, axis=1),
        y.max(axis=1),
        np.std(y, axis=1),
    ]
    return np.vstack(aggregated_features)


def compute_spectral_features(y: np.ndarray) -> np.ndarray:
    features = np.vstack(
        [
            compute_spectral_centroid(y),
            compute_spectral_bandwidth(y),
            compute_spectral_rolloff(y),
            compute_spectral_contrast(y),  # 7
            compute_spectral_flatness(y),
        ]
    )
    return aggregate(features)
