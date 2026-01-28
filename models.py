import os
from typing import Callable
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# from tensorflow.keras.applications.resnet import preprocess_input
# from tensorflow.keras.applications.densenet import preprocess_input
from features import (
    norm,
    compute_tempo_features,
    compute_mel_spectrogram,
    compute_chromagram,
    compute_spectral_features,
)


if not len(tf.config.list_physical_devices("GPU")):
    print("No GPU found. Using CPU.")

from data import get_train_test_val_features_and_targets, unique_genres, n_genres
from utils import LivePlot

os.makedirs("imgs", exist_ok=True)


def majority_vote_by_id(y_true, y_pred, ids):
    """
    calculates the accuracy using average prediction across music tracks
    """
    id_to_true = {}
    id_to_preds = {}

    for yt, yp, i in zip(y_true, y_pred, ids):
        id_to_preds.setdefault(i, []).append(yp)
        id_to_true[i] = yt  # same label for all segments of an id

    y_true_ids = []
    y_pred_ids = []

    for i in id_to_preds:
        y_true_ids.append(id_to_true[i])
        y_pred_ids.append(Counter(id_to_preds[i]).most_common(1)[0][0])

    return np.mean(np.array(y_true_ids) == np.array(y_pred_ids))


def pipeline(
    df: pd.DataFrame,
    train_size: float,
    test_size: float,
    splits: int,
    feature_extractor: Callable,
    model_creator: Callable,
    epochs: int,
    batch_size: int,
    earlystop_patience: int,
    learning_rate: float,
    augmented: bool,
    plot_confusion_matrix: bool = True,
    liveplot_training: bool = False,
):
    (
        (X_train, y_train, ids_train),
        (X_test, y_test, ids_test),
        (X_val, y_val, ids_val),
    ) = get_train_test_val_features_and_targets(
        df=df,
        feature_extractor=feature_extractor,
        splits=splits,
        train_size=train_size,
        test_size=test_size,
        augmented=augmented,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(train_size, float)
    assert isinstance(test_size, float)
    assert isinstance(splits, int)
    assert isinstance(epochs, int)
    assert isinstance(batch_size, int)
    assert isinstance(earlystop_patience, int)
    assert isinstance(learning_rate, float)
    assert isinstance(augmented, bool)
    assert isinstance(plot_confusion_matrix, bool)
    assert isinstance(liveplot_training, bool)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_val) == len(y_val)
    assert len(X_train) + len(X_test) + len(X_val) == len(df) * splits

    def assert_scaling(X: np.ndarray) -> bool:
        assert X.max() <= 255
        assert X.min() >= 0

    # don't normalize for ResNet and DenseNet since we use preprocess_input
    if model_creator not in [create_resnet, create_densenet]:
        (X_train, X_test, X_val) = norm(
            X_train=X_train,
            X_test=X_test,
            X_val=X_val,
            feature_extractor=feature_extractor,
        )
    elif feature_extractor == compute_mel_spectrogram and model_creator in [
        create_resnet,
        create_densenet,
    ]:
        # mel-spectrogram needs rescaling from [-80,0] to [0,255]
        X_train = (X_train + 80.0) * (255.0 / 80.0)
        X_test = (X_test + 80.0) * (255.0 / 80.0)
        X_val = (X_val + 80.0) * (255.0 / 80.0)

        assert_scaling(X_train)
        assert_scaling(X_test)
        assert_scaling(X_val)

    elif feature_extractor == compute_chromagram and model_creator in [
        create_resnet,
        create_densenet,
    ]:
        X_train *= 255.0
        X_test *= 255.0
        X_val *= 255.0
        assert_scaling(X_train)
        assert_scaling(X_test)
        assert_scaling(X_val)

    # create model
    model = model_creator(X=X_train, learning_rate=learning_rate)

    is_sklearn_model = isinstance(model, sklearn.svm._classes.SVC) or isinstance(
        model, sklearn.ensemble._forest.RandomForestClassifier
    )

    # these models require flattened inputs
    if is_sklearn_model or (model_creator == create_simple_feedforward_model):
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    if is_sklearn_model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        history, loss = None, None
    else:
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=earlystop_patience,
                restore_best_weights=True,
            )
        ]
        if liveplot_training:
            callbacks.append(LivePlot(logy=True))
        with tf.device("/GPU:0"):
            # fit model
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0,
            )

            # calculate loss & accuracy
            logits = model.predict(X_test, batch_size=batch_size)
            y_pred = np.argmax(logits, axis=1)

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            loss = loss_fn(y_test, logits).numpy()

    segment_accuracy = tf.reduce_mean(tf.cast(y_pred == y_test, tf.float32)).numpy()
    track_accuracy = majority_vote_by_id(y_test, y_pred, ids_test)
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_genres,
        yticklabels=unique_genres,
    )

    plt.title(
        f"Confusion Matrix. Segment Accuracy={segment_accuracy*100:.2f}% Track Accuracy={track_accuracy*100:.2f}%"
    )
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.savefig(
        f"imgs/feature_extractor={feature_extractor.__name__}"
        + f"_model_creator={model_creator.__name__}_epochs_{epochs}"
        + f"_batch_size={batch_size}_learning_rate={learning_rate}.png"
    )

    if plot_confusion_matrix:
        plt.show()
    else:
        plt.close()
    return model, history, loss, segment_accuracy, track_accuracy, recall, f1


def compile(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],  # loss is implicitly included
    )
    model.summary()
    return model


def _create_dense_or_resnet_model(
    X: np.ndarray, learning_rate: float, base_model: tf.keras.Model
) -> tf.keras.Model:
    input_shape = (*X.shape[1:], 1)  # (12, 431, 1)
    inputs = layers.Input(shape=input_shape)

    # Resize height to meet ResNet constraints
    x = layers.Resizing(224, 224)(inputs)

    # Adapter to 3 channels
    x = layers.Conv2D(3, (1, 1), padding="same", trainable=False)(x)
    if base_model == tf.keras.applications.ResNet50V2:
        x = tf.keras.applications.resnet.preprocess_input(x)
    elif base_model == tf.keras.applications.DenseNet121:
        x = tf.keras.applications.densenet.preprocess_input(x)
    else:
        raise ValueError("Unsupported base model")

    # Pretrained ResNet
    base_model = base_model(
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    for layer in base_model.layers[-10:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(n_genres, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return compile(model, learning_rate)


def create_densenet(X: np.ndarray, learning_rate: float) -> tf.keras.Model:
    return _create_dense_or_resnet_model(
        X=X, learning_rate=learning_rate, base_model=tf.keras.applications.DenseNet121
    )


def create_resnet(X: np.ndarray, learning_rate: float) -> tf.keras.Model:
    return _create_dense_or_resnet_model(
        X=X, learning_rate=learning_rate, base_model=tf.keras.applications.ResNet50V2
    )


def create_simple_feedforward_model(
    X: np.ndarray, learning_rate: float
) -> tf.keras.Model:
    """Simple MLP model"""
    n_features = np.prod(X.shape[1:])
    inp = tf.keras.layers.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(1024, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    out = tf.keras.layers.Dense(n_genres, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return compile(model, learning_rate)


def create_residual_cnn_model(X: np.ndarray, learning_rate: float) -> tf.keras.Model:
    # this needs 2d input (+ batch dim)
    l2_reg = regularizers.l2(1e-4)  # L2 weight decay

    inp = layers.Input(shape=(X.shape[1], X.shape[2], 1))  # add channel dim

    x = inp if len(X.shape) == 3 else x

    # CNN BLOCKS
    # block 1
    conv1 = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
    )(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.MaxPooling2D((2, 2))(conv1)

    # block 2
    conv2 = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
    )(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.MaxPooling2D((2, 2))(conv2)

    # block 3
    conv3 = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
    )(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.MaxPooling2D((2, 2))(conv3)

    # flatten CNN output
    x = layers.GlobalAveragePooling2D()(conv3)

    # DENSE LAYERS
    # block 1
    dense = layers.Dense(512, activation="relu", kernel_regularizer=l2_reg)(x)
    dense = layers.Dropout(0.5)(dense)
    res = layers.Dense(512, activation=None, kernel_regularizer=l2_reg)(x)
    x = layers.Add()([dense, res])
    x = layers.Activation("relu")(x)

    # block 2
    dense = layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(x)
    dense = layers.Dropout(0.5)(dense)
    res = layers.Dense(256, activation=None, kernel_regularizer=l2_reg)(x)
    x = layers.Add()([dense, res])
    x = layers.Activation("relu")(x)

    # block 3
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(n_genres, activation="softmax", kernel_regularizer=l2_reg)(x)
    model = models.Model(inputs=inp, outputs=out)
    return compile(model, learning_rate)


def create_complex_feedforward_model(
    X: np.ndarray, learning_rate: float
) -> tf.keras.Model:
    l2_reg = tf.keras.regularizers.l2(1e-4)

    if len(X.shape) == 3:
        n_features = np.prod(X.shape[1:])
        inp = tf.keras.layers.Input(shape=(X.shape[1], X.shape[2]))
        x = tf.keras.layers.Flatten()(inp)
    else:
        n_features = X.shape[1]
        inp = tf.keras.layers.Input(shape=(n_features,))
        x = inp

    dense = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2_reg)(x)
    dense = tf.keras.layers.Dropout(0.5)(dense)
    res = tf.keras.layers.Dense(1024, activation=None, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.Add()([dense, res])
    x = tf.keras.layers.Activation("relu")(x)

    dense = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2_reg)(x)
    dense = tf.keras.layers.Dropout(0.5)(dense)
    res = tf.keras.layers.Dense(1024, activation=None, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.Add()([dense, res])
    x = tf.keras.layers.Activation("relu")(x)

    dense = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=l2_reg)(x)
    dense = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=l2_reg)(
        dense
    )
    dense = tf.keras.layers.Dropout(0.5)(dense)
    res = tf.keras.layers.Dense(512, activation=None, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.Add()([dense, res])
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)

    # Output layer
    out = tf.keras.layers.Dense(
        n_genres, activation="softmax", kernel_regularizer=l2_reg
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    return compile(model, learning_rate)


def create_svm(**kwargs) -> svm.SVC:
    return svm.SVC()


def create_random_forest(
    n_estimators: int = 100,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_depth: int = None,
    **kwargs,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        max_depth=max_depth,
    )
