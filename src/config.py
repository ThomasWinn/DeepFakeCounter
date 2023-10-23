# Data
DATASET_DIR = '../dataset'
CACHE_FILE = 'data/mean_and_std.pt'
BATCH_SIZE = 32
NUM_WORKERS = 0
VALID_SIZE = 0.2

# Model
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0

# Trainer
ACCELERATOR = 'gpu' # device whether gpu, cpu, tpu, etc.
LOG_DIR = '../' # location to place tensorboard_logs
DEVICES = 1 # how many gpus
MIN_EPOCHS = 1
MAX_EPOCHS = 100
