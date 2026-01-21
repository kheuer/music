# audio parameters
sample_rate = 44100
clips_per_file = 6  # each file of 30 seconds is split into this many clips.
seconds_per_clip = 30 / clips_per_file
assert seconds_per_clip >= 2
assert not (
    661560 / clips_per_file % 10
)  # ensure that each sampled clip can be cut off cleanly

# dataset split parameters
test_size = 0.2
val_size = 0.2
train_size = 1 - (test_size + val_size)
assert round(test_size + val_size + train_size, 6) == 1.0
assert all(size >= 0 for size in [test_size, val_size, train_size])
