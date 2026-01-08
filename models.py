from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import layers, models

if not len(tf.config.list_physical_devices("GPU")):
    print("No GPU found. Using CPU.")

from data import get_train_test_val_features_and_targets, unique_genres, n_genres
from utils import LivePlot


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
    learning_rate: int,
):
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = (
        get_train_test_val_features_and_targets(
            df=df,
            feature_extractor=feature_extractor,
            splits=splits,
            train_size=train_size,
            test_size=test_size,
        )
    )
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_val) == len(y_val)
    assert len(X_train) + len(X_test) + len(X_val) == len(df) * splits
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Validation size: {len(X_val)}")

    scaler = StandardScaler().fit(X_train)

    x_train_transformed = scaler.transform(X_train)
    x_val_transformed = scaler.transform(X_val)
    x_test_transformed = scaler.transform(X_test)

    x_train_transformed = tf.convert_to_tensor(x_train_transformed, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

    # create model
    model = model_creator(n_features=x_train_transformed.shape[1])

    with tf.device("/GPU:0"):
        # fit model
        history = model.fit(
            x_train_transformed,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val_transformed, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=earlystop_patience,
                    restore_best_weights=True,
                ),
                LivePlot(logy=True),
            ],
            verbose=0,
        )

    # calculate loss & accuracy
    logits = model(x_test_transformed)
    y_pred = np.argmax(logits, axis=1)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(y_test, logits).numpy()

    accuracy = tf.reduce_mean(tf.cast(y_pred == y_test, tf.float32)).numpy()

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

    plt.title(f"Confusion Matrix. Accuracy={accuracy*100:.2f}%")
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.show()
    return model, history, loss, accuracy


def compile(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],  # loss is implicitly included
    )
    model.summary()
    return model


def create_densenet(input_shape: tuple, learning_rate: float) -> tf.keras.Model:
    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        name="densenet201",
    )
    base_model.trainable = False
    output = layers.Dense(8, activation="softmax")(base_model.output)
    model = models.Model(inputs=base_model.input, outputs=output)
    return compile(model, learning_rate)


def create_resnet(input_shape: tuple, learning_rate: float) -> tf.keras.Model:
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        name="resnet50v2_base",
    )
    base_model.trainable = False
    output = layers.Dense(8, activation="softmax")(base_model.output)
    model = models.Model(inputs=base_model.input, outputs=output)
    return compile(model, learning_rate)


def create_simple_feedforward_model(
    n_features: int, learning_rate: float
) -> tf.keras.Model:
    """Simple MLP model"""
    inp = tf.keras.layers.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(n_genres, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return compile(model, learning_rate)


def create_svm() -> svm.SVC:
    return svm.SVC()


def create_random_forest(
    n_estimators: int = 100, min_samples_split: int = 2, min_samples_leaf: int = 1
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
