# =============================================================================
# train.py — R2-Former v2
# =============================================================================
# WHAT THIS FILE DOES (plain English):
#
# This file runs the actual TRAINING of the R2-Former v2 model.
# Training means: show the model thousands of labelled examples, let it make
# predictions, measure how wrong it is (the "loss"), and nudge the model's
# weights slightly in the direction that reduces the error. Repeat many times.
#
# THE TRAINING LOOP (what happens every epoch):
#   1. TRAIN PHASE   — show every training example once, update weights
#   2. VALIDATE PHASE — check performance on unseen validation examples
#   3. SAVE           — if this epoch is the best so far, save the model
#   4. LOG            — record all metrics to a CSV file for the paper
#
# KEY CONCEPTS EXPLAINED:
#
#   Loss        — a number measuring how wrong the model is. Lower = better.
#                 We use CrossEntropyLoss (standard for classification).
#
#   Accuracy    — percentage of examples predicted correctly.
#
#   Optimiser   — the algorithm that updates the model weights.
#                 We use AdamW (Adam with Weight Decay), standard for transformers.
#
#   Scheduler   — gradually changes the learning rate during training.
#                 We use linear warmup then linear decay — standard for BERT-style models.
#
#   Epoch       — one full pass through the entire training dataset.
#                 We train for NUM_EPOCHS = 5 epochs.
#
# OUTPUT FILES (saved to results/ and logs/):
#   best_model.pt        — the model checkpoint with best validation accuracy
#   training_log.csv     — epoch-by-epoch metrics for the paper
# =============================================================================

import os                              # File path operations
import time                            # Timing training duration
import csv                             # Saving metrics to CSV
import torch                           # PyTorch
import torch.nn as nn                  # Neural network tools
from torch.optim import AdamW          # AdamW optimiser
from transformers import get_linear_schedule_with_warmup   # LR scheduler
from tqdm import tqdm                  # Progress bars in terminal

from config import *                   # All hyperparameters
from data_pipeline import get_data_pipeline   # Data loading
from model import build_model          # Model construction


# =============================================================================
# HELPER — COMPUTE ACCURACY
# =============================================================================

def compute_accuracy(logits, labels):
    """
    Compute the fraction of predictions that match the true labels.

    Args:
        logits : [batch, num_classes] — raw model output scores
        labels : [batch]             — true class indices (0 or 1)

    Returns:
        accuracy : float in [0, 1]
    """
    # torch.argmax returns the index of the highest score (= predicted class)
    predictions = torch.argmax(logits, dim=-1)   # [batch]
    correct     = (predictions == labels).sum().item()
    total       = labels.size(0)
    return correct / total


# =============================================================================
# TRAINING PHASE — one epoch
# =============================================================================
# WHAT HAPPENS EACH TRAINING STEP:
#
#   1. Load a batch of 16 examples from the DataLoader
#   2. Move tensors to MPS (GPU)
#   3. Zero out gradients from previous step
#   4. Forward pass: model predicts → logits + gate_value
#   5. Compute loss: how wrong is the prediction?
#   6. Backward pass: compute gradients (how to fix the weights)
#   7. Gradient clipping: cap gradient magnitude to prevent exploding gradients
#   8. Optimiser step: update the weights
#   9. Scheduler step: adjust learning rate
#  10. Record loss and accuracy
# =============================================================================

def train_one_epoch(model, loader, optimiser, scheduler, criterion, device, epoch):
    """
    Run one full training epoch.

    Args:
        model     : R2FormerV2
        loader    : training DataLoader
        optimiser : AdamW
        scheduler : linear warmup scheduler
        criterion : CrossEntropyLoss
        device    : torch.device (mps)
        epoch     : int — current epoch number (for display)

    Returns:
        avg_loss : float — mean loss over the epoch
        avg_acc  : float — mean accuracy over the epoch
    """

    model.train()   # Switch to training mode (enables dropout, batch norm updates)

    total_loss = 0.0
    total_acc  = 0.0
    num_batches = len(loader)

    # tqdm wraps the DataLoader and shows a live progress bar in terminal
    progress = tqdm(
        loader,
        desc    = f"  Epoch {epoch} [TRAIN]",
        leave   = True,
        ncols   = 100,
    )

    for batch_idx, batch in enumerate(progress):

        # ── MOVE DATA TO DEVICE ───────────────────────────────────────────────
        # Data comes off the DataLoader on CPU. We move it to MPS (Apple GPU).
        input_ids      = batch['input_ids'].to(device)       # [batch, seq_len]
        attention_mask = batch['attention_mask'].to(device)  # [batch, seq_len]
        labels         = batch['label'].to(device)           # [batch]

        # ── ZERO GRADIENTS ────────────────────────────────────────────────────
        # PyTorch accumulates gradients by default. We must reset them each step
        # or they'd add up across batches, corrupting the weight updates.
        optimiser.zero_grad()

        # ── FORWARD PASS ──────────────────────────────────────────────────────
        # Run the input through the full R2-Former v2 pipeline.
        # logits shape: [batch, 2]
        # gate_value shape: [batch, 1] (we log this but don't use it for loss)
        logits, gate_value = model(input_ids, attention_mask)

        # ── COMPUTE LOSS ──────────────────────────────────────────────────────
        # CrossEntropyLoss compares our raw scores (logits) to the true labels.
        # Internally it applies softmax + negative log likelihood.
        # A perfect prediction gives loss ≈ 0; random guessing gives loss ≈ 0.69
        loss = criterion(logits, labels)

        # ── BACKWARD PASS ─────────────────────────────────────────────────────
        # Computes how much each weight contributed to the loss.
        # This fills the .grad attribute of every parameter.
        loss.backward()

        # ── GRADIENT CLIPPING ─────────────────────────────────────────────────
        # If gradients get too large ("exploding gradients"), training can
        # become unstable. Clipping to max_norm=1.0 is standard for BERT models.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── UPDATE WEIGHTS ────────────────────────────────────────────────────
        # AdamW uses the gradients to update every weight in the model.
        optimiser.step()

        # ── UPDATE LEARNING RATE ──────────────────────────────────────────────
        # The scheduler adjusts the learning rate according to its schedule.
        scheduler.step()

        # ── RECORD METRICS ────────────────────────────────────────────────────
        batch_loss = loss.item()                             # scalar float
        batch_acc  = compute_accuracy(logits, labels)
        total_loss += batch_loss
        total_acc  += batch_acc

        # Update the live progress bar with current loss and accuracy
        progress.set_postfix({
            'loss' : f"{batch_loss:.4f}",
            'acc'  : f"{batch_acc:.3f}",
            'lr'   : f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_loss = total_loss / num_batches
    avg_acc  = total_acc  / num_batches
    return avg_loss, avg_acc


# =============================================================================
# VALIDATION PHASE — one epoch
# =============================================================================
# Validation is like training but WITHOUT updating weights.
# We just measure performance on data the model has never seen during training.
#
# KEY DIFFERENCES from training:
#   • model.eval()         — disables dropout (we want deterministic output)
#   • torch.no_grad()      — don't compute gradients (saves memory, 2x faster)
#   • No optimiser.step()  — we never update weights during validation
# =============================================================================

def validate_one_epoch(model, loader, criterion, device, epoch):
    """
    Run one full validation epoch (no weight updates).

    Args:
        model     : R2FormerV2
        loader    : validation DataLoader
        criterion : CrossEntropyLoss
        device    : torch.device
        epoch     : int — current epoch number

    Returns:
        avg_loss     : float — mean validation loss
        avg_acc      : float — mean validation accuracy
        avg_gate     : float — mean confidence gate value (for analysis)
    """

    model.eval()   # Disable dropout

    total_loss = 0.0
    total_acc  = 0.0
    total_gate = 0.0
    num_batches = len(loader)

    progress = tqdm(
        loader,
        desc  = f"  Epoch {epoch} [VAL]  ",
        leave = True,
        ncols = 100,
    )

    with torch.no_grad():   # No gradient computation needed
        for batch in progress:

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits, gate_value = model(input_ids, attention_mask)

            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc  += compute_accuracy(logits, labels)
            total_gate += gate_value.mean().item()   # average gate across batch

            progress.set_postfix({
                'loss' : f"{loss.item():.4f}",
                'acc'  : f"{compute_accuracy(logits, labels):.3f}",
            })

    avg_loss = total_loss / num_batches
    avg_acc  = total_acc  / num_batches
    avg_gate = total_gate / num_batches
    return avg_loss, avg_acc, avg_gate


# =============================================================================
# SAVE CHECKPOINT
# =============================================================================
# We save the model whenever it achieves a new best validation accuracy.
# This means at the end of training we always have the BEST model, not just
# the most recent one (which might have slightly overfit on the last epoch).
# =============================================================================

def save_checkpoint(model, optimiser, epoch, val_acc, path):
    """
    Save model weights and training state to disk.

    Args:
        model     : R2FormerV2
        optimiser : AdamW (saved so training could be resumed)
        epoch     : int — which epoch this checkpoint is from
        val_acc   : float — validation accuracy at this checkpoint
        path      : str — file path to save to
    """
    checkpoint = {
        'epoch'            : epoch,
        'model_state_dict' : model.state_dict(),
        'optimiser_state_dict' : optimiser.state_dict(),
        'val_acc'          : val_acc,
    }
    torch.save(checkpoint, path)
    print(f"\n  [✓] New best model saved → {path}  (val_acc={val_acc:.4f})")


# =============================================================================
# LOGGING — save metrics to CSV
# =============================================================================
# We write a CSV file after every epoch so you can track training progress
# and use the numbers directly in the paper's results table.
# =============================================================================

def log_metrics(log_path, epoch, train_loss, train_acc, val_loss, val_acc, avg_gate, elapsed):
    """
    Append one row of metrics to the training log CSV.

    Args:
        log_path   : str — path to CSV file
        epoch      : int
        train_loss : float
        train_acc  : float
        val_loss   : float
        val_acc    : float
        avg_gate   : float — mean confidence gate value this epoch
        elapsed    : float — seconds this epoch took
    """
    # If the file doesn't exist yet, write the header row first
    write_header = not os.path.exists(log_path)

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'epoch', 'train_loss', 'train_acc',
                'val_loss', 'val_acc', 'avg_gate_value', 'epoch_time_s'
            ])
        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{avg_gate:.6f}",
            f"{elapsed:.1f}",
        ])


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """
    Full training pipeline for R2-Former v2.

    Steps:
        1. Set up paths and seed
        2. Load data pipeline
        3. Build model
        4. Set up optimiser and scheduler
        5. Run NUM_EPOCHS training + validation epochs
        6. Save best model and log metrics
    """

    # ── PATHS ─────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs",    exist_ok=True)
    checkpoint_path = os.path.join("results", "best_model.pt")
    log_path        = os.path.join("logs",    "training_log.csv")

    # Remove old log so we start fresh (don't append to a previous run)
    if os.path.exists(log_path):
        os.remove(log_path)

    print("=" * 60)
    print("  R2-FORMER v2 — TRAINING")
    print("=" * 60)
    print(f"\n  Device        : {DEVICE}")
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Warmup steps  : {WARMUP_STEPS}")
    print(f"  Weight decay  : {WEIGHT_DECAY}")
    print(f"  Seed          : {SEED}")

    # ── SEED ──────────────────────────────────────────────────────────────────
    # Import and call set_seed here too, to guarantee reproducibility
    # even if train.py is run without running data_pipeline.py first.
    import random, numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

    # ── DATA ──────────────────────────────────────────────────────────────────
    print(f"\n[Train] Loading data pipeline...")
    train_loader, val_loader, test_loader, tokenizer = get_data_pipeline()

    # ── MODEL ─────────────────────────────────────────────────────────────────
    model = build_model(DEVICE)

    # ── LOSS FUNCTION ─────────────────────────────────────────────────────────
    # CrossEntropyLoss is the standard loss for classification.
    # It combines LogSoftmax + NLLLoss internally.
    # For a correct prediction with high confidence, loss ≈ 0.
    # For a wrong prediction with high confidence, loss can be very large.
    criterion = nn.CrossEntropyLoss()

    # ── OPTIMISER: AdamW ──────────────────────────────────────────────────────
    # WHAT IS AdamW?
    # Adam adjusts the learning rate per-parameter based on gradient history.
    # The "W" means "Weight Decay" — a regularisation penalty that keeps weights
    # small and helps prevent overfitting.
    #
    # We apply weight decay ONLY to weight matrices, NOT to biases or layer norms.
    # This is standard practice for BERT-style fine-tuning.
    #
    # WHY DIFFERENT LEARNING RATES?
    # The pre-trained backbone (DistilBERT) should be updated gently —
    # it already "knows" language, we don't want to destroy that knowledge.
    # The new modules (rationale extractor, gate, refinement) train from scratch
    # and can handle a slightly higher effective rate via weight decay separation.
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimiser_grouped_parameters = [
        {
            # Parameters that SHOULD have weight decay (weight matrices)
            'params'      : [p for n, p in model.named_parameters()
                             if not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY,
        },
        {
            # Parameters that should NOT have weight decay (biases, layer norms)
            'params'      : [p for n, p in model.named_parameters()
                             if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    optimiser = AdamW(optimiser_grouped_parameters, lr=LEARNING_RATE)

    # ── LEARNING RATE SCHEDULER ───────────────────────────────────────────────
    # WHAT IS A SCHEDULER?
    # Instead of a fixed learning rate throughout training, we use:
    #
    #   Phase 1 — WARMUP (first WARMUP_STEPS steps):
    #     LR rises linearly from 0 → LEARNING_RATE.
    #     Why? Starting with a large LR on random weights can destabilise training.
    #     Warming up slowly gives the model a stable start.
    #
    #   Phase 2 — LINEAR DECAY (remaining steps):
    #     LR falls linearly from LEARNING_RATE → 0.
    #     Why? As the model gets better, smaller updates prevent overshooting.
    #
    # total_steps = number of batches × number of epochs
    total_steps = len(train_loader) * NUM_EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps   = WARMUP_STEPS,
        num_training_steps = total_steps,
    )

    print(f"\n[Train] Total training steps : {total_steps:,}")
    print(f"[Train] Warmup steps         : {WARMUP_STEPS}")
    print(f"[Train] Steps per epoch      : {len(train_loader):,}")

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    training_start = time.time()

    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING — {NUM_EPOCHS} EPOCHS")
    print(f"{'='*60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"{'─'*60}")

        # ── TRAIN ─────────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimiser, scheduler, criterion, DEVICE, epoch
        )

        # ── VALIDATE ──────────────────────────────────────────────────────────
        val_loss, val_acc, avg_gate = validate_one_epoch(
            model, val_loader, criterion, DEVICE, epoch
        )

        epoch_elapsed = time.time() - epoch_start

        # ── PRINT EPOCH SUMMARY ───────────────────────────────────────────────
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train  — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  ({train_acc*100:.2f}%)")
        print(f"    Val    — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  ({val_acc*100:.2f}%)")
        print(f"    Avg confidence gate : {avg_gate:.4f}")
        print(f"    Time   : {epoch_elapsed:.1f}s")

        # ── SAVE BEST MODEL ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimiser, epoch, val_acc, checkpoint_path)

        # ── LOG METRICS ───────────────────────────────────────────────────────
        log_metrics(
            log_path, epoch,
            train_loss, train_acc,
            val_loss, val_acc,
            avg_gate, epoch_elapsed,
        )

    # ── TRAINING COMPLETE ─────────────────────────────────────────────────────
    total_elapsed = time.time() - training_start
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed  % 60)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best validation accuracy : {best_val_acc*100:.2f}%")
    print(f"  Best model saved to      : {checkpoint_path}")
    print(f"  Training log saved to    : {log_path}")
    print(f"  Total training time      : {minutes}m {seconds}s")
    print(f"\n[✓] Ready for evaluate.py")

    return best_val_acc


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train()