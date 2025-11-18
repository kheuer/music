import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import params

random.seed(42)
np.random.seed(42)


def load_file(file_path):
    audio, _ = librosa.load(file_path, sr=params.sample_rate)
    return audio


if __name__ == "__main__":
    # delete old cache
    try:
        os.remove("data.csv")
    except FileNotFoundError:
        pass
    try:
        os.remove("audio.npy")
    except FileNotFoundError:
        pass

    # load metadata
    tracks = pd.read_csv("data/fma_metadata/tracks.csv", index_col=0, header=[0, 1])

    # create a dataframe with just those files that are part of the small subset
    builder = {"path": [], "genre": [], "track_id": []}

    for folder in os.listdir("data/fma_small"):
        if not os.path.isdir(os.path.join("data/fma_small", folder)):
            continue
        for file in os.listdir(os.path.join("data/fma_small", folder)):
            if not file.endswith(".mp3"):
                continue
            # iterate over all existing files (we only use a subset of the full dataset)
            # and add them to our dataframe along with their genre
            track_id = int(file.removesuffix(".mp3"))
            genre = tracks.loc[track_id, ("track", "genre_top")]
            builder["track_id"].append(track_id)
            builder["path"].append(os.path.join("data/fma_small", folder, file))
            builder["genre"].append(genre)

    df = pd.DataFrame(builder)
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv("data.csv", index=False)

    del builder

    # sanity checks
    assert all([l > 950 for l in df["genre"].value_counts().values])
    assert len(df) > 950 * 8
    assert len(df["genre"].unique()) == 8

    # open the audio files and pad them to the same length
    builder = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio files"):
        builder.append(load_file(row["path"]))

    max_len = max(len(signal) for signal in builder)

    padded_signals = np.zeros((len(builder), max_len))

    for i, audio_signal in enumerate(builder):
        padded_signals[i, : len(audio_signal)] = audio_signal

    np.save("audio.npy", padded_signals)

    assert padded_signals.shape == (len(df), max_len)
    assert max_len >= params.sample_rate * 30  # at least 30 seconds

    print("done")
