import os
import itertools
import pandas as pd
from tqdm import tqdm

from features import (
    compute_chromagram,
    compute_mel_spectrogram,
    compute_tempo_features,
    compute_spectral_features,
)
from models import (
    pipeline,
    create_simple_feedforward_model,
    create_complex_feedforward_model,
    create_residual_cnn_model,
    create_densenet,
    create_resnet,
    create_random_forest,
    create_svm,
)
from data import df

splits_list = [10]
batch_sizes = [256]
learning_rates = [0.0001]
feature_extractors = [
    # compute_spectral_features,
    # compute_tempo_features,
    compute_chromagram,
    compute_mel_spectrogram,
]
model_creators = [
    # create_random_forest,
    # create_svm,
    # create_simple_feedforward_model,
    # create_resnet,
    create_densenet,
    create_complex_feedforward_model,
    create_residual_cnn_model,
]

augmentations = [False, True]
combinations = list(
    itertools.product(
        augmentations,
        splits_list,
        batch_sizes,
        learning_rates,
        feature_extractors,
        model_creators,
    )
)

output_file = "grid_search_results.csv"

if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
else:
    results_df = pd.DataFrame(
        columns=[
            "augmented",
            "splits",
            "batch_size",
            "learning_rate",
            "feature_extractor",
            "model_creator",
            "loss",
            "segment_accuracy",
            "track_accuracy",
            "recall",
            "f1",
        ]
    )

completed = set(
    tuple(x)
    for x in results_df[
        [
            "augmented",
            "splits",
            "batch_size",
            "learning_rate",
            "feature_extractor",
            "model_creator",
        ]
    ].values
)


unsaved_combinations = []
for augmented, splits, batch_size, lr, feature_extractor, model_creator in combinations:
    if (
        augmented,
        splits,
        batch_size,
        lr,
        feature_extractor.__name__,
        model_creator.__name__,
    ) in completed:
        continue
    elif (
        model_creator in (create_random_forest, create_svm)
        and (batch_size != batch_sizes[0])
        and (learning_rates != learning_rates[0])
    ):
        # these models ignore batch size and learning rate, no need to recalculate those combinations
        continue

    unsaved_combinations.append(
        (augmented, splits, batch_size, lr, feature_extractor, model_creator)
    )


for augmented, splits, batch_size, lr, feature_extractor, model_creator in tqdm(
    unsaved_combinations, total=len(unsaved_combinations), desc="Gridsearch Loop"
):

    print(
        f"Running: augmented={augmented} splits={splits}, batch_size={batch_size}, lr={lr}, feature_extractor={feature_extractor.__name__}, model_creator={model_creator.__name__}"
    )
    model, history, loss, segment_accuracy, track_accuracy, recall, f1 = pipeline(
        df=df,
        train_size=0.8,
        test_size=0.1,
        splits=splits,
        feature_extractor=feature_extractor,
        model_creator=model_creator,
        epochs=70,
        batch_size=batch_size,
        earlystop_patience=15,
        learning_rate=lr,
        augmented=augmented,
        plot_confusion_matrix=False,
        liveplot_training=False,
    )

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    {
                        "augmented": augmented,
                        "splits": splits,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "feature_extractor": feature_extractor.__name__,
                        "model_creator": model_creator.__name__,
                        "loss": loss,
                        "segment_accuracy": segment_accuracy,
                        "track_accuracy": track_accuracy,
                        "recall": recall,
                        "f1": f1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    results_df.to_csv(output_file, index=False)
    results_df.to_excel(output_file.replace(".csv", ".xlsx"), index=False)
