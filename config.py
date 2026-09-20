# config.py
# ============================================================
# R2-Former v2 — Configuration File
# All hyperparameters and settings in one place
# ============================================================

import torch

# ── Device ──────────────────────────────────────────────────
# Automatically uses M2 GPU (MPS) if available, else CPU
DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ── Model ────────────────────────────────────────────────────
# DistilBERT is a lightweight, fast version of BERT
# Perfect for M2 MacBook — same quality, half the size
PRETRAINED_MODEL   = "distilbert-base-uncased"
NUM_CLASSES        = 2          # Positive or Negative sentiment
MAX_SEQ_LENGTH     = 128        # Maximum number of tokens per sentence
HIDDEN_SIZE        = 768        # DistilBERT's internal representation size
NUM_RATIONALE_TOKENS = 10       # How many key tokens to extract as rationale

# ── Dual Rationale Extraction ────────────────────────────────
LOCAL_ATTN_HEADS   = 4          # Attention heads for local (word-level) rationale
GLOBAL_ATTN_HEADS  = 4          # Attention heads for global (sentence-level) rationale

# ── Refinement Transformer ───────────────────────────────────
REFINEMENT_LAYERS  = 2          # Number of iterative refinement passes
REFINEMENT_HEADS   = 8          # Attention heads in refinement transformer
DROPOUT_RATE       = 0.1        # Dropout for regularisation (prevents overfitting)

# ── Training ─────────────────────────────────────────────────
BATCH_SIZE         = 32         # Samples per training step (safe for M2 memory)
LEARNING_RATE      = 2e-5       # How fast the model learns (standard for BERT)
NUM_EPOCHS         = 5          # Number of full passes through training data
WARMUP_STEPS       = 100        # Gradual learning rate warmup at start
WEIGHT_DECAY       = 0.01       # Regularisation to prevent overfitting

# ── Dataset ──────────────────────────────────────────────────
DATASET_NAME       = "sst2"     # SST-2 Stanford Sentiment Treebank
DATASET_CONFIG     = "default"
TRAIN_SPLIT        = "train"
VAL_SPLIT          = "validation"
TEST_SPLIT         = "validation"  # SST-2 uses validation as test

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR           = "data/"
MODEL_SAVE_PATH    = "models/r2former_v2_best.pt"
RESULTS_DIR        = "results/"
LOGS_DIR           = "logs/"

# ── Reproducibility ───────────────────────────────────────────
SEED               = 42         # Fixed seed so results are reproducible

print(f"Config loaded. Device: {DEVICE}")