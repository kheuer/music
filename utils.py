import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import tensorflow as tf
import librosa
import params
from IPython.display import clear_output

from data import load_file, n_genres, df


class LivePlot(tf.keras.callbacks.Callback):
    def __init__(self, logy=False):
        super().__init__()
        self.logy = logy
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Collect metrics
        self.train_loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        self.train_acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

        # Update plot
        clear_output(wait=True)
        plt.figure(figsize=(12, 5))

        # loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss" + " (Log Scale)" if self.logy else "Loss")
        if self.logy:
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both")

        # accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label="Train Accuracy")
        plt.plot(self.val_acc, label="Validation Accuracy")
        plt.axhline(
            y=1 / n_genres,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Random Threshold",
        )
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def show_mel_spectrogram(index: int):
    mel_spectrogram = librosa.feature.mfcc(
        y=load_file(df.loc[index, "path"]), sr=params.sample_rate, n_mels=40
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram_db,
        sr=params.sample_rate,
        x_axis="time",
        y_axis="mel",
        cmap="coolwarm",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.title(
        f'Mel Spectrogram for audio at index {index}, genre: {df.iloc[index]["genre"]}'
    )
    plt.show()


def play_audio(index: int):
    ipd.display(
        ipd.Audio(data=load_file(df.loc[index, "path"]), rate=params.sample_rate)
    )


def investigate(index: int):
    show_mel_spectrogram(index)
    play_audio(index)
