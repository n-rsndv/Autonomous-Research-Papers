# =============================================================================
# evaluate.py — R2-Former v2
# =============================================================================
# WHAT THIS FILE DOES (plain English):
#
# Training tells us the model learned something. Evaluation tells us EXACTLY
# how well it learned, with proper metrics that go in the paper.
#
# This file:
#   1. Loads the best saved model (results/best_model.pt)
#   2. Runs it on the validation set (no weight updates — pure testing)
#   3. Computes four standard NLP metrics:
#        • Accuracy   — overall % correct
#        • Precision  — of all "positive" predictions, how many were right?
#        • Recall     — of all actual positives, how many did we catch?
#        • F1 Score   — harmonic mean of precision and recall (main paper metric)
#   4. Builds a confusion matrix (shows exactly where the model makes mistakes)
#   5. Analyses the confidence gate values (novel contribution evidence)
#   6. Saves a full results summary to results/evaluation_report.txt
#
# WHY THESE METRICS?
#
#   Accuracy alone can be misleading if classes are imbalanced.
#   SST-2 is roughly balanced (50% pos / 50% neg), so accuracy is fine here,
#   but F1 is the standard metric reported in NLP papers for GLUE benchmarks.
#
# WHAT IS A CONFUSION MATRIX?
#
#   A 2×2 table showing:
#                    Predicted NEG    Predicted POS
#   Actual NEG    [  True Neg (TN)    False Pos (FP) ]
#   Actual POS    [  False Neg (FN)   True Pos (TP)  ]
#
#   Perfect model: all examples on the diagonal (TN and TP).
#   Off-diagonal entries are mistakes.
# =============================================================================

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from config import *
from data_pipeline import get_data_pipeline
from model import build_model


# =============================================================================
# LOAD THE SAVED MODEL
# =============================================================================

def load_best_model(checkpoint_path, device):
    """
    Load the best model checkpoint saved during training.

    Args:
        checkpoint_path : str — path to best_model.pt
        device          : torch.device

    Returns:
        model : R2FormerV2 with loaded weights
        epoch : int — which epoch this checkpoint came from
        val_acc : float — validation accuracy at save time
    """
    print(f"[Evaluate] Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}\n"
            f"Please run train.py first."
        )

    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build a fresh model and load the saved weights into it
    model = build_model(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch   = checkpoint['epoch']
    val_acc = checkpoint['val_acc']

    print(f"[Evaluate] Loaded model from epoch {epoch} (val_acc={val_acc:.4f})")
    return model, epoch, val_acc


# =============================================================================
# RUN INFERENCE — collect all predictions and labels
# =============================================================================

def run_inference(model, loader, device):
    """
    Run the model on an entire DataLoader and collect predictions.

    Args:
        model  : R2FormerV2
        loader : DataLoader (val or test)
        device : torch.device

    Returns:
        all_preds  : list of predicted class indices (0 or 1)
        all_labels : list of true class indices (0 or 1)
        all_gates  : list of confidence gate values (floats)
        all_logits : list of raw logit pairs [neg_score, pos_score]
    """

    model.eval()   # Disable dropout

    all_preds  = []
    all_labels = []
    all_gates  = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Running inference", ncols=80):

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            # Forward pass
            logits, gate_value = model(input_ids, attention_mask)

            # Get predicted class (index of highest logit)
            preds = torch.argmax(logits, dim=-1)

            # Move everything to CPU and convert to Python lists
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_gates.extend(gate_value.squeeze(1).cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())

    return all_preds, all_labels, all_gates, all_logits


# =============================================================================
# COMPUTE METRICS
# =============================================================================

def compute_metrics(all_preds, all_labels):
    """
    Compute all evaluation metrics from predictions and true labels.

    Args:
        all_preds  : list of int — predicted classes
        all_labels : list of int — true classes

    Returns:
        metrics : dict with accuracy, precision, recall, f1, confusion_matrix
    """

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Precision, Recall, F1 — computed per class then macro-averaged
    # 'macro' means: compute metric for each class separately, then average.
    # This treats both classes equally regardless of how many examples each has.
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        average = 'macro',
        zero_division = 0,
    )

    # Per-class metrics (negative and positive separately)
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        all_labels, all_preds,
        average = None,
        zero_division = 0,
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Full classification report (formatted string for the report file)
    report = classification_report(
        all_labels, all_preds,
        target_names = ['NEGATIVE', 'POSITIVE'],
        digits       = 4,
    )

    return {
        'accuracy'       : accuracy,
        'precision'      : precision,
        'recall'         : recall,
        'f1'             : f1,
        'precision_per'  : precision_per,
        'recall_per'     : recall_per,
        'f1_per'         : f1_per,
        'support_per'    : support_per,
        'confusion_matrix': cm,
        'report'         : report,
    }


# =============================================================================
# ANALYSE CONFIDENCE GATE
# =============================================================================
# This is a novel analysis we can include in the paper to justify the
# Adaptive Confidence Gate contribution.
#
# HYPOTHESIS: Examples where the model is WRONG should have LOWER gate values
# (less confident) than examples where the model is RIGHT.
# If this holds, it validates that the gate is genuinely measuring confidence.
# =============================================================================

def analyse_confidence_gate(all_preds, all_labels, all_gates):
    """
    Analyse whether the confidence gate correlates with correctness.

    Returns:
        gate_analysis : dict with mean gate values for correct vs wrong predictions
    """

    correct_gates = [g for p, l, g in zip(all_preds, all_labels, all_gates) if p == l]
    wrong_gates   = [g for p, l, g in zip(all_preds, all_labels, all_gates) if p != l]

    mean_gate_correct = np.mean(correct_gates) if correct_gates else 0.0
    mean_gate_wrong   = np.mean(wrong_gates)   if wrong_gates   else 0.0

    # Gate value distribution
    gate_array = np.array(all_gates)

    return {
        'mean_gate_overall'  : float(np.mean(gate_array)),
        'std_gate_overall'   : float(np.std(gate_array)),
        'mean_gate_correct'  : mean_gate_correct,
        'mean_gate_wrong'    : mean_gate_wrong,
        'n_correct'          : len(correct_gates),
        'n_wrong'            : len(wrong_gates),
        'gate_min'           : float(np.min(gate_array)),
        'gate_max'           : float(np.max(gate_array)),
    }


# =============================================================================
# PRINT AND SAVE RESULTS
# =============================================================================

def print_and_save_results(metrics, gate_analysis, best_epoch, split_name, report_path):
    """
    Print a full evaluation report to terminal and save to file.

    Args:
        metrics       : dict from compute_metrics()
        gate_analysis : dict from analyse_confidence_gate()
        best_epoch    : int — which training epoch the model is from
        split_name    : str — 'validation' or 'test'
        report_path   : str — path to save the report
    """

    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()

    lines = []
    lines.append("=" * 62)
    lines.append("  R2-FORMER v2 — EVALUATION REPORT")
    lines.append(f"  Split: {split_name.upper()}   |   Best epoch: {best_epoch}")
    lines.append("=" * 62)

    lines.append("\n── MAIN METRICS ─────────────────────────────────────────────")
    lines.append(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    lines.append(f"  F1 Score  : {metrics['f1']*100:.2f}%   ← primary paper metric")
    lines.append(f"  Precision : {metrics['precision']*100:.2f}%")
    lines.append(f"  Recall    : {metrics['recall']*100:.2f}%")

    lines.append("\n── PER-CLASS METRICS ────────────────────────────────────────")
    lines.append(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append(f"  {'─'*52}")
    class_names = ['NEGATIVE', 'POSITIVE']
    for i, name in enumerate(class_names):
        lines.append(
            f"  {name:<12} "
            f"{metrics['precision_per'][i]*100:>9.2f}% "
            f"{metrics['recall_per'][i]*100:>9.2f}% "
            f"{metrics['f1_per'][i]*100:>9.2f}% "
            f"{int(metrics['support_per'][i]):>10}"
        )

    lines.append("\n── CONFUSION MATRIX ─────────────────────────────────────────")
    lines.append("                  Predicted NEG    Predicted POS")
    lines.append(f"  Actual NEG   [  {tn:>8}          {fp:>8}    ]")
    lines.append(f"  Actual POS   [  {fn:>8}          {tp:>8}    ]")
    lines.append(f"\n  True Negatives  (TN): {tn}  — correctly predicted NEGATIVE")
    lines.append(f"  True Positives  (TP): {tp}  — correctly predicted POSITIVE")
    lines.append(f"  False Positives (FP): {fp}  — predicted POS, actually NEG")
    lines.append(f"  False Negatives (FN): {fn}  — predicted NEG, actually POS")

    lines.append("\n── CONFIDENCE GATE ANALYSIS ─────────────────────────────────")
    lines.append(f"  Mean gate (overall)        : {gate_analysis['mean_gate_overall']:.4f}")
    lines.append(f"  Std  gate (overall)        : {gate_analysis['std_gate_overall']:.4f}")
    lines.append(f"  Mean gate (correct preds)  : {gate_analysis['mean_gate_correct']:.4f}")
    lines.append(f"  Mean gate (wrong preds)    : {gate_analysis['mean_gate_wrong']:.4f}")
    lines.append(f"  Gate range                 : [{gate_analysis['gate_min']:.4f}, {gate_analysis['gate_max']:.4f}]")

    # Check if hypothesis holds (correct preds have higher gate = more confident)
    diff = gate_analysis['mean_gate_correct'] - gate_analysis['mean_gate_wrong']
    if diff > 0:
        lines.append(f"\n  ✓ HYPOTHESIS CONFIRMED: correct predictions have higher")
        lines.append(f"    gate values by {diff:.4f} — gate genuinely measures confidence.")
        lines.append(f"    This supports the Adaptive Confidence Gate contribution.")
    else:
        lines.append(f"\n  Gate difference: {diff:.4f} (gate vs correctness correlation weak)")

    lines.append("\n── FULL CLASSIFICATION REPORT ───────────────────────────────")
    lines.append(metrics['report'])

    lines.append("=" * 62)
    lines.append("  FILES SAVED")
    lines.append(f"  Report : {report_path}")
    lines.append("=" * 62)

    # Print to terminal
    full_text = "\n".join(lines)
    print(full_text)

    # Save to file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(full_text)

    print(f"\n[✓] Report saved to {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def evaluate():
    """
    Full evaluation pipeline. Loads best model, runs on validation set,
    computes and saves all metrics.
    """

    print("=" * 62)
    print("  R2-FORMER v2 — EVALUATION")
    print("=" * 62)

    # ── PATHS ─────────────────────────────────────────────────────
    checkpoint_path = os.path.join("results", "best_model.pt")
    report_path     = os.path.join("results", "evaluation_report.txt")

    # ── DATA ──────────────────────────────────────────────────────
    print("\n[Evaluate] Loading data pipeline...")
    _, val_loader, _, _ = get_data_pipeline()

    # ── MODEL ─────────────────────────────────────────────────────
    model, best_epoch, saved_val_acc = load_best_model(checkpoint_path, DEVICE)

    # ── INFERENCE ─────────────────────────────────────────────────
    print("\n[Evaluate] Running inference on validation set...")
    all_preds, all_labels, all_gates, all_logits = run_inference(
        model, val_loader, DEVICE
    )

    # ── METRICS ───────────────────────────────────────────────────
    print("\n[Evaluate] Computing metrics...")
    metrics      = compute_metrics(all_preds, all_labels)
    gate_analysis = analyse_confidence_gate(all_preds, all_labels, all_gates)

    # ── REPORT ────────────────────────────────────────────────────
    print()
    print_and_save_results(
        metrics, gate_analysis, best_epoch,
        split_name  = "validation",
        report_path = report_path,
    )

    return metrics['accuracy'], metrics['f1']


if __name__ == "__main__":
    evaluate()