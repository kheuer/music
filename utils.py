import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import tensorflow as tf
import librosa
import params
from IPython.display import clear_output

from data import load_file, n_genres, df

matplotlib.use("TkAgg")  # or "Qt5Agg"


class LivePlot(tf.keras.callbacks.Callback):
    def __init__(self, logy=False):
        """
        Args:
            logy: use log scale for loss
            show: whether to display plots live
        """
        super().__init__()
        self.logy = logy

        # metric history
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        # figure
        self.fig = None
        self.axes = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Collect metrics
        self.train_loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        self.train_acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

        # create figure once
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
            plt.ion()  # interactive mode for terminal

        # loss
        ax_loss = self.axes[0]
        ax_loss.clear()
        ax_loss.plot(self.train_loss, label="Train Loss")
        ax_loss.plot(self.val_loss, label="Validation Loss")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss" + (" (Log Scale)" if self.logy else ""))
        if self.logy:
            ax_loss.set_yscale("log")
        ax_loss.legend()
        ax_loss.grid(True, which="both")

        # accuracy
        ax_acc = self.axes[1]
        ax_acc.clear()
        ax_acc.plot(self.train_acc, label="Train Segment Accuracy")
        ax_acc.plot(self.val_acc, label="Validation Segment Accuracy")
        ax_acc.axhline(
            y=1 / n_genres,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Random Threshold",
        )
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True)
        ax_acc.set_ylim(0, 1)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  # refresh without blocking

        # show the figure once, then only update
        if epoch == 0:
            plt.show(block=False)


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
