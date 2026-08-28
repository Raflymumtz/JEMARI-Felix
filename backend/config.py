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
CONFUSION_PATH = os.path.join(SAVED_MODEL_DIR, "confusion_matrix.csv")

# --- image / hand-crop settings ---
# Crops are *cached* at CACHE_IMG_SIZE by preprocess.py and randomly
# cropped/resized down to IMG_SIZE during training, so scale augmentation
# has real pixels to zoom into instead of upsampling a blurry thumbnail.
CACHE_IMG_SIZE = 160
IMG_SIZE = 112          # spatial resolution fed into the CNN backbone
HAND_MARGIN = 0.45      # extra margin (fraction of bbox size) around detected hand
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- hand landmarks (auxiliary input stream) ---
N_LANDMARKS = 21
LANDMARK_DIM = N_LANDMARKS * 3 + 1  # 63 coords + 1 validity flag

# --- temporal / sequence settings (Transformer input) ---
WINDOW = 8              # number of consecutive frames per sequence sample
STRIDE = 2              # stride between windows when building training sequences

# --- model architecture ---
BACKBONE = "mobilenet_v3_small"   # or "mobilenet_v2" for the ablation table
CNN_FEATURE_DIM = 576   # MobileNetV3-Small features output channels
D_CNN = 128             # CNN feature projected down to this
D_LM = 64               # landmark branch output width
D_MODEL = 192           # fused token width fed to the Transformer
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 384
DROPOUT = 0.2

# --- training ---
BATCH_SIZE = 64
EPOCHS = 40
LR_BACKBONE = 1e-4      # pretrained ImageNet features: fine-tune gently
LR_HEAD = 1e-3          # everything trained from scratch
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999
PATIENCE = 8
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42


# Session-splitting thresholds are derived from WINDOW/VAL_SPLIT/TEST_SPLIT
# by dataset.min_splittable_group() rather than hard-coded here.

# Max dHash bit distance at which two frames count as the same picture.
# Near-duplicates are forced into the same split so the test set can never
# contain a variant of a training image (see dataset._quarantine_duplicates).
# Deliberately one step wider than audit_dataset.py's default check: the
# quarantine should always be a superset of what the audit looks for.
DUP_HAMMING = 6

# --- real-time inference / debounce ---
VOTE_WINDOW = 5          # how many recent predictions to look at for stability
VOTE_MIN_AGREEMENT = 3   # how many of them must agree before committing a letter
CONF_THRESHOLD = 0.55
NO_HAND_FRAMES_FOR_SPACE = 12  # consecutive no-hand frames -> insert a space
