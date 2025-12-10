### Datasplit (duration of samples) from SmallFMA
- 3 Sekunden
- 5 Sekunden
- 10 Sekunden
- 30 Sekunden
### Features
- static feature vector (5*6) (centroid, bandwidth, rolloff, contrast, flatness) -> mean, std dev, min, max, unteres quartil, oberes quartil
- complete feature vector (centroid, bandwidth, contrast, flatness, (tempo))
- MelSpectogram
- Chromagram
- Tempogram
### Models / Architecture
- SVM
- Simple feedforward 
- DenseNet
- ResNet
