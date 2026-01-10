import os
import itertools
import pandas as pd

from features import (
    compute_chromagram,
    compute_mel_spectrogram,
    compute_tempo_features,
    compute_spectral_features,
)
from models import (
    pipeline,
    create_simple_feedforward_model,
    create_densenet,
    create_resnet,
)
from data import df

splits_list = [10, 6, 3, 1]
batch_sizes = [16, 32, 64, 128, 256]
learning_rates = [0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07]
feature_extractors = [
    compute_spectral_features,
    compute_tempo_features,
    compute_mel_spectrogram,
    compute_chromagram,
]
model_creators = [
    create_resnet,
    create_densenet,
    create_simple_feedforward_model,
]

combinations = list(
    itertools.product(
        splits_list, batch_sizes, learning_rates, feature_extractors, model_creators
    )
)

output_file = "grid_search_results.csv"

if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
else:
    results_df = pd.DataFrame(
        columns=[
            "splits",
            "batch_size",
            "learning_rate",
            "feature_extractor",
            "model_creator",
            "accuracy",
            "loss",
        ]
    )

completed = set(
    tuple(x)
    for x in results_df[
        ["splits", "batch_size", "learning_rate", "feature_extractor", "model_creator"]
    ].values
)

for splits, batch_size, lr, feature_extractor, model_creator in combinations:
    if (
        splits,
        batch_size,
        lr,
        feature_extractor.__name__,
        model_creator.__name__,
    ) in completed:
        continue

    print(f"Running: splits={splits}, batch_size={batch_size}, lr={lr}")
    model, history, loss, accuracy = pipeline(
        df=df,
        train_size=0.6,
        test_size=0.2,
        splits=splits,
        feature_extractor=feature_extractor,
        model_creator=model_creator,
        epochs=100,
        batch_size=batch_size,
        earlystop_patience=10,
        learning_rate=lr,
    )

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    {
                        "splits": splits,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "feature_extractor": feature_extractor.__name__,
                        "model_creator": model_creator.__name__,
                        "loss": loss,
                        "accuracy": accuracy,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    results_df.to_csv(output_file, index=False)
