from data import *
from features import *
from models import *

model, history, loss, segment_accuracy, track_accuracy, recall, f1 = pipeline(
    df=df,
    train_size=0.6,
    test_size=0.2,
    splits=10,
    feature_extractor=compute_mel_spectrogram,
    model_creator=create_residual_cnn_model,
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
