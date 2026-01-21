import random
import os
from typing import Callable
import hashlib
import inspect
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import params
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


random.seed(42)
np.random.seed(42)

try:
    os.makedirs("cache")
except FileExistsError:
    pass


def load_file(file_path: str) -> np.ndarray:
    # Load file
    audio, _ = librosa.load(file_path, sr=params.sample_rate)

    target_length = params.sample_rate * 30

    # zero padding
    if len(audio) < target_length:
        padded_audio = np.pad(audio, (0, target_length - len(audio)), "constant")
    else:
        padded_audio = audio[:target_length]

    augment = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        ]
    )
    padded_audio = augment(samples=padded_audio, sample_rate=params.sample_rate)

    assert len(padded_audio) == 661500
    return padded_audio


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

# build the dataframe
df = pd.DataFrame(builder)
del builder

# create a consistent random order
permutation = np.load("permutation.npy")
valid_permutation = permutation[permutation < len(df)]

df = df.iloc[valid_permutation].reset_index(drop=True)

# sanity checks
assert all([l > 950 for l in df["genre"].value_counts().values])
assert len(df) > 950 * 8
assert len(df["genre"].unique()) == 8

# manipulate
unique_genres = sorted(list(set(df["genre"])))
n_genres = len(unique_genres)
assert n_genres == 8
genre_to_id = {unique_genres[_]: _ for _ in range(n_genres)}
id_to_genre = {_: unique_genres[_] for _ in range(n_genres)}
df["genre_id"] = df["genre"].map(genre_to_id)


### Cache Management


def compute_hash(data) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(data).encode("utf-8"))
    return sha256_hash.hexdigest()


def split_dataset(
    df: pd.DataFrame, train_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # split
    train_size = int(len(df) * train_size)
    test_size = int(len(df) * test_size)
    val_size = len(df) - train_size - test_size
    assert train_size > 10
    assert test_size > 10
    assert val_size > 10
    assert len(df) == train_size + test_size + val_size

    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size : train_size + test_size]
    df_val = df.iloc[train_size + test_size :]

    assert len(df) == len(df_train) + len(df_test) + len(df_val)
    return df_train, df_test, df_val


def build_targets_features(
    data: pd.DataFrame, splits: int, feature_extractor: Callable, desc=str
) -> tuple[np.ndarray, np.ndarray]:

    features = []
    targets = []
    ids = []
    for _, row in tqdm(
        data.iterrows(), total=len(data), desc=f"Building {desc} dataset"
    ):
        raw = load_file(row["path"])
        assert not len(raw) % splits
        step = int(len(raw) / splits)
        for i in range(splits):
            raw_chunk = raw[i * step : (i + 1) * step]
            features.append(feature_extractor(raw_chunk))
            targets.append(row["genre_id"])
            ids.append(row["track_id"])

    X = np.stack(features)
    y = np.array(targets).flatten()
    ids = np.array(ids)

    assert len(X) == len(data) * splits
    assert len(y) == len(data) * splits
    idx = np.random.permutation(len(X))
    return X[idx], y[idx], ids[idx]


def get_train_test_val_features_and_targets(
    df: pd.DataFrame,
    feature_extractor: Callable,
    splits: int,
    train_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_file(df.loc[0, "path"])
    source_code = inspect.getsource(feature_extractor)
    identifier = compute_hash(
        (
            feature_extractor(data),
            feature_extractor.__name__,
            source_code,
            splits,
            train_size,
            test_size,
            len(df),
        )
    )

    if os.path.isdir(os.path.join("cache", identifier)):
        return (
            (
                np.load(os.path.join("cache", identifier, "X_train.npy")),
                np.load(os.path.join("cache", identifier, "y_train.npy")),
                np.load(os.path.join("cache", identifier, "ids_train.npy")),
            ),
            (
                np.load(os.path.join("cache", identifier, "X_test.npy")),
                np.load(os.path.join("cache", identifier, "y_test.npy")),
                np.load(os.path.join("cache", identifier, "ids_test.npy")),
            ),
            (
                np.load(os.path.join("cache", identifier, "X_val.npy")),
                np.load(os.path.join("cache", identifier, "y_val.npy")),
                np.load(os.path.join("cache", identifier, "ids_val.npy")),
            ),
        )

    else:
        df_train, df_test, df_val = split_dataset(
            df=df, train_size=train_size, test_size=test_size
        )

        # create data for model
        X_train, y_train, ids_train = build_targets_features(
            data=df_train,
            splits=splits,
            feature_extractor=feature_extractor,
            desc="train",
        )
        X_test, y_test, ids_test = build_targets_features(
            data=df_test,
            splits=splits,
            feature_extractor=feature_extractor,
            desc="test",
        )
        X_val, y_val, ids_val = build_targets_features(
            data=df_val, splits=splits, feature_extractor=feature_extractor, desc="val"
        )
        assert len(X_train) + len(X_test) + len(X_val) == len(df) * splits
        assert len(y_train) + len(y_test) + len(y_val) == len(df) * splits

        # cache for future use
        os.makedirs(os.path.join("cache", identifier))
        np.save(os.path.join("cache", identifier, "X_train.npy"), X_train)
        np.save(os.path.join("cache", identifier, "y_train.npy"), y_train)
        np.save(os.path.join("cache", identifier, "ids_train.npy"), ids_train)
        np.save(os.path.join("cache", identifier, "X_test.npy"), X_test)
        np.save(os.path.join("cache", identifier, "y_test.npy"), y_test)
        np.save(os.path.join("cache", identifier, "ids_test.npy"), ids_test)
        np.save(os.path.join("cache", identifier, "X_val.npy"), X_val)
        np.save(os.path.join("cache", identifier, "y_val.npy"), y_val)
        np.save(os.path.join("cache", identifier, "ids_val.npy"), ids_val)

        return (
            (X_train, y_train, ids_train),
            (X_test, y_test, ids_test),
            (X_val, y_val, ids_val),
        )
