### Datasplit (duration of samples) from SmallFMA
- 3 Sekunden
- 5 Sekunden
- 10 Sekunden
- 30 Sekunden
### Features
- spectral feature vector (11*7) (centroid, bandwidth, rolloff, contrast, flatness) -> mean, median, std dev, min, max, unteres quartil, oberes quartil
- tempo feature vector (7*time frame) -> mean, median, std dev, min, max, unteres quartil, oberes quartil
- MelSpectogram
- Chromagram
### Models / Architecture
- Traditional: SVM, Random Forest
- Simple feedforward 
- DenseNet
