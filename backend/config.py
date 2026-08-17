"""Shared configuration for the BISINDO real-time translation system.

Values here are referenced by preprocessing, training and the inference
server so the three stages always agree on image size, window length, etc.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "DATASET FELIX")
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "PROCESSED")

SAVED_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.pt")
LABEL_MAP_PATH = os.path.join(SAVED_MODEL_DIR, "label_map.json")
METRICS_PATH = os.path.join(SAVED_MODEL_DIR, "metrics.json")
HISTORY_PATH = os.path.join(SAVED_MODEL_DIR, "history.csv")

# --- image / hand-crop settings ---
IMG_SIZE = 112          # spatial resolution fed into the CNN backbone
HAND_MARGIN = 0.35      # extra margin (fraction of bbox size) around detected hand
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- temporal / sequence settings (Transformer input) ---
WINDOW = 8              # number of consecutive frames per sequence sample
STRIDE = 4              # stride between windows when building training sequences

# --- model architecture ---
CNN_FEATURE_DIM = 1280  # MobileNetV2 output feature dimension
D_MODEL = 192
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 384
DROPOUT = 0.2

# --- training ---
BATCH_SIZE = 24
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# --- real-time inference / debounce ---
VOTE_WINDOW = 5          # how many recent predictions to look at for stability
VOTE_MIN_AGREEMENT = 3   # how many of them must agree before committing a letter
CONF_THRESHOLD = 0.55
NO_HAND_FRAMES_FOR_SPACE = 12  # consecutive no-hand frames -> insert a space
