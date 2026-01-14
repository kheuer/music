from data import df
from features import (
    compute_chromagram,
    compute_mel_spectrogram,
    compute_spectral_features,
    compute_tempo_features,
)
from models import (
    pipeline,
    create_densenet,
    create_resnet,
    create_residual_cnn_model,
    create_simple_feedforward_model,
    create_complex_feedforward_model,
    create_svm,
    create_random_forest,
)


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

print(
    "segment_accuracy:",
    segment_accuracy,
    "track_accuracy:",
    track_accuracy,
    "recall",
    recall,
    "f1",
    f1,
)
input()
