# Datasplit (duration of samples) from SmallFMA
- can be defined by "splits" parameter, (e.g. splits = 10 -> duration of sample = 30/10 -> 3 seconds)

# Two setups:
## Basic Approach:
### Features:
- tempo feature vector (7,) -> mean, median, std dev, min, max, unteres quartil, oberes quartil -> compute_tempo_features
- spectral feature vector (7,11) (centroid, bandwidth, rolloff, contrast, flatness) -> mean, median, std dev, min, max, unteres quartil, oberes quartil -> compute_static_features
### Models / Architecture:
- SVM, Random Forest
- Simple feedforward
- Complex feedforward
## Deep Learning Approach:
### Features:
- MelSpectogram -> compute_mel_spectogram
- Chromagram -> compute_chromagram
### Models / Architecture:
- DenseNet201
- ResNet50v2
- Complex feedforward
- Residual CNN