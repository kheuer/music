### Datasplit (duration of samples) from SmallFMA
- 3 Sekunden
- 5 Sekunden
- 10 Sekunden
- 30 Sekunden
### Features
- spectral feature vector (5*6) (centroid, bandwidth, rolloff, contrast, flatness) -> mean, std dev, min, max, unteres quartil, oberes quartil
- tempo feature vector (5*6) (centroid (tempo), bandwidth, contrast, flatness) -> erst mitteln dann Features
- (complete feature vector (centroid, bandwidth, contrast, flatness, (tempo)))
- MelSpectogram
- Chromagram
### Models / Architecture
- Traditional: SVM, Random Forest
- Simple feedforward 
- DenseNet
- (ResNet)
