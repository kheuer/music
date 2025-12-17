# Datasplit (duration of samples) from SmallFMA
- 3 Sekunden
- 5 Sekunden
- 10 Sekunden
- 30 Sekunden

# Two setups:
## Basic Approach:
### Features:
- spectral feature vector (7,11) (centroid, bandwidth, rolloff, contrast, flatness) -> mean, median, std dev, min, max, unteres quartil, oberes quartil -> compute_static_features
- tempo feature vector (7,time frame) -> mean, median, std dev, min, max, unteres quartil, oberes quartil -> compute_tempo_features
### Models / Architecture:
- SVM, Random Forest
- Simple feedforward
## Deep Learning Approach:
### Features:
- MelSpectogram (min(n_mels,20),time frame) -> compute_mel_spectogram
- Chromagram (12,time frame) -> compute_chromagram
### Models / Architecture:
- DenseNet201
- ResNet50v2
