# audio parameters
sample_rate = 22050

# dataset split parameters
test_size = 0.2
val_size = 0.2
train_size = 1 - (test_size + val_size)
assert round(test_size + val_size + train_size, 6) == 1.0
assert all(size >= 0 for size in [test_size, val_size, train_size])
