# Music Genre Classification

For this project, Python 3.10 is required.

## Environment Setup

To set up a Python environment, install Anaconda, navigate to the project directory, and execute:

```
conda create --name music python=3.10
conda activate music
pip install -r requirements.txt
sh data/download.sh
```

During development, dependencies may be added. To update them, simply re-run:

```
pip install -r requirements.txt
```

### Windows Notes

On Windows, you can execute the download script using Git Bash (must be installed):

```python
import subprocess
bash_path = r"C:\Program Files\Git\git-bash.exe" # installation path of git-bash.exe
sh_file = r"...\music\data\download.sh"
```

If you encounter issues loading audio files (e.g. `NoBackendError` from `librosa`), install the required backends:

```
conda install -c conda-forge ffmpeg audioread -y
```
















## Usage: Running Experiments

All experiments are executed via the `experiment.py` script. This script runs the full training and evaluation pipeline, including dataset splits, feature extraction, model selection, and training configuration.

### Basic Command

After activating the environment and downloading the data, run:

```bash
python experiment.py
```

### Experiment Configuration

Experiments are configured directly inside `experiment.py` via the `pipeline(...)` call. The most important parameters are:

* **df**: Loaded dataset (imported from `data`)
* **train_size**: Fraction of data used for training (e.g. `0.6`)
* **test_size**: Fraction of data used for testing (e.g. `0.2`) (validation size is inferred)
* **splits**: Number of audio samples to create from each 30 second audio clip in the dataset
* **feature_extractor**: Audio feature function, one of:

  * `compute_mel_spectrogram`
  * `compute_chromagram`
  * `compute_spectral_features`
  * `compute_tempo_features`
* **model_creator**: Model factory function, one of:

  * `create_densenet`
  * `create_resnet`
  * `create_residual_cnn_model`
  * `create_simple_feedforward_model`
  * `create_complex_feedforward_model`
  * `create_svm`
  * `create_random_forest`
* **epochs**: Maximum number of training epochs
* **batch_size**: Training batch size
* **earlystop_patience**: Early stopping patience
* **learning_rate**: Optimizer learning rate
* **plot_confusion_matrix**: Whether to show a confusion matrix after evaluation
* **liveplot_training**: Whether to plot training metrics live

### Example

```python
model, history, loss, segment_accuracy, track_accuracy, recall, f1 = pipeline(
    df=df,
    train_size=0.6,
    test_size=0.2,
    splits=3,
    feature_extractor=compute_mel_spectrogram,
    model_creator=create_densenet,
    epochs=100,
    batch_size=32,
    earlystop_patience=10,
    learning_rate=0.0001,
    plot_confusion_matrix=True,
    liveplot_training=True,
)
```

### Outputs

At the end of an experiment, the following metrics are printed:

* **Segment accuracy**: Accuracy over individual audio segments
* **Track accuracy**: Accuracy aggregated at track level
* **Recall**: Macro recall over genres
* **F1 score**: Macro F1 score

The script pauses at the end (`input()`) to keep plots open until user confirmation.

To run a different experiment, simply modify `experiment.py` and re-run the script.

Not all model architecures and feature extractions functions are compatible with each other. The compatibilities are outlined in `approaches.md`.