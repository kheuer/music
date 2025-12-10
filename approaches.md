### Datasplit (duration of samples) from SmallFMA
- 3 Sekunden
- 5 Sekunden
- 10 Sekunden
- 30 Sekunden
### Features
- static feature vector (24 (4*6) + 1) (centroid, bandwidth, contrast, flatness, (tempo)) -> mean, std dev, min, max, unteres quartil, oberes quartil
- complete feature vector (centroid, bandwidth, contrast, flatness, (tempo))
- MelSpectogram
- Chromagram
- Tempogram
### Models / Architecture
- SVM
- Simple feedforward 
- DenseNet
- ResNet
