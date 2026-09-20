# =============================================================================
# data_pipeline.py — R2-Former v2
# =============================================================================
# WHAT THIS FILE DOES (plain English):
#
# Before training a model, we need to prepare the raw text data so the model
# can understand it. Computers don't understand words — they understand numbers.
# This file handles the full journey from raw text → numbers the model can use.
#
# The pipeline has 4 stages:
#   1. SET SEED      — Fix randomness so results are reproducible
#   2. LOAD DATA     — Download SST-2 dataset from Hugging Face
#   3. TOKENISE      — Convert text → token IDs (numbers) using DistilBERT's vocabulary
#   4. DATALOADER    — Bundle data into batches for efficient training
#
# OUTPUT: Three DataLoader objects (train / validation / test) that train.py will use.
# =============================================================================

import random                          # Python's built-in random number generator
import numpy as np                     # Numerical computing library
import torch                           # PyTorch — our deep learning framework
from torch.utils.data import DataLoader, Dataset   # Tools for batching data
from datasets import load_dataset      # Hugging Face library to download datasets
from transformers import AutoTokenizer  # Loads the DistilBERT tokeniser
from config import *                   # Import all settings from config.py (SEED, BATCH_SIZE, etc.)


# =============================================================================
# STEP 1 — SET SEED FOR REPRODUCIBILITY
# =============================================================================
# WHY THIS MATTERS:
# Deep learning involves lots of random operations (weight initialisation,
# data shuffling, dropout). If we don't fix the seed, every run gives slightly
# different results, making it impossible to compare experiments or reproduce
# results for a research paper.
#
# By setting the same seed everywhere (Python, NumPy, PyTorch, and the GPU),
# we guarantee that if you run this script twice with the same settings,
# you get exactly the same numbers.
# =============================================================================

def set_seed(seed: int = SEED):
    """
    Fix all random number generators to the same seed value.
    Call this at the very start of any script.
    """
    random.seed(seed)                        # Python's random module
    np.random.seed(seed)                     # NumPy's random module
    torch.manual_seed(seed)                  # PyTorch CPU operations
    torch.cuda.manual_seed_all(seed)         # PyTorch GPU operations (CUDA)

    # For M2 Mac (MPS backend) — sets seed for Apple Silicon GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"[Seed] All random seeds fixed to: {seed}")


# =============================================================================
# STEP 2 — LOAD AND INSPECT THE SST-2 DATASET
# =============================================================================
# WHAT IS SST-2?
# SST-2 (Stanford Sentiment Treebank, binary) is one of the most well-known
# benchmarks for sentiment analysis. It contains movie review sentences
# labelled as either:
#   • 0 = Negative  (e.g. "This film was a complete waste of time.")
#   • 1 = Positive  (e.g. "A masterpiece of modern cinema.")
#
# The dataset has ~67,000 training examples and ~872 validation examples.
# It lives on the Hugging Face Hub and downloads automatically on first run.
#
# NOTE: SST-2 is part of the GLUE benchmark. Its test split has NO labels
# (they're hidden for the leaderboard), so we evaluate on the validation split.
# =============================================================================

def load_sst2_dataset():
    """
    Download and return the SST-2 dataset from Hugging Face.

    Returns:
        dataset_dict: A dictionary with keys 'train', 'validation', 'test'
                      Each value is a list of {'sentence': str, 'label': int} examples
    """
    print("[Data] Downloading SST-2 dataset from Hugging Face Hub...")
    print("       (This may take a minute on first run — it caches locally afterwards)")

    # load_dataset("stanfordnlp/sst2") fetches the SST-2 subset of the GLUE benchmark
    dataset = load_dataset("stanfordnlp/sst2")

    # Show a quick summary so you can see what was loaded
    print(f"\n[Data] Dataset loaded successfully!")
    print(f"       Training examples  : {len(dataset['train'])}")
    print(f"       Validation examples: {len(dataset['validation'])}")
    print(f"       Test examples      : {len(dataset['test'])}")

    # Show two sample examples so you can inspect the raw data
    print(f"\n[Data] Sample training examples:")
    for i in range(2):
        ex = dataset['train'][i]
        label_word = "POSITIVE" if ex['label'] == 1 else "NEGATIVE"
        print(f"       [{i}] Label={label_word}  Text: \"{ex['sentence'][:80]}...\"")

    return dataset


# =============================================================================
# STEP 3 — TOKENISATION
# =============================================================================
# WHAT IS TOKENISATION? (Plain English Explanation)
#
# DistilBERT doesn't read English — it reads integers from a fixed vocabulary
# of 30,522 tokens. A "token" is roughly a word or word-piece.
#
# Example:
#   Raw text  : "The food was surprisingly good!"
#   Tokens    : ['the', 'food', 'was', 'surprising', '##ly', 'good', '!']
#   Token IDs : [101, 1996, 2833, 2001, 10773, 2135, 2204, 999, 102]
#               ^^^                                                  ^^^
#             [CLS] special start token                [SEP] special end token
#
# The tokeniser also creates two additional arrays per sentence:
#   • attention_mask : 1 where there is a real token, 0 where there is padding
#   • token_type_ids : Used in BERT for two-sentence tasks (always 0 here)
#
# WHY MAX_SEQ_LENGTH=128?
# Transformer attention scales quadratically with sequence length (O(n²)).
# Most SST-2 sentences are short, so 128 tokens captures ~99% of them.
# Shorter = faster training, less memory.
# =============================================================================

def load_tokenizer():
    """
    Load the DistilBERT tokeniser.
    The tokeniser converts text strings into integer token IDs.

    Returns:
        tokenizer: A Hugging Face tokeniser object
    """
    print(f"\n[Tokeniser] Loading '{PRETRAINED_MODEL}' tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    print(f"[Tokeniser] Vocabulary size: {tokenizer.vocab_size:,} tokens")
    return tokenizer


# =============================================================================
# STEP 4 — CUSTOM PYTORCH DATASET CLASS
# =============================================================================
# WHAT IS A PYTORCH DATASET?
#
# PyTorch requires data to be wrapped in a "Dataset" object. Think of it like
# a smart list that knows:
#   1. How many items it contains  (__len__)
#   2. How to return item number N  (__getitem__)
#
# The DataLoader (Step 5) will call __getitem__ repeatedly to build batches.
#
# Our SST2Dataset does the tokenisation ON THE FLY for each example when it's
# requested. This is memory-efficient — we never store all tokenised tensors
# at once, only the current batch.
# =============================================================================

class SST2Dataset(Dataset):
    """
    A PyTorch Dataset that wraps SST-2 examples.

    Each item returned is a dictionary with:
        input_ids      : Tensor of shape [MAX_SEQ_LENGTH] — the token ID integers
        attention_mask : Tensor of shape [MAX_SEQ_LENGTH] — 1=real token, 0=padding
        label          : Scalar tensor — 0 (negative) or 1 (positive)
    """

    def __init__(self, hf_split, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        """
        Initialise the dataset.

        Args:
            hf_split   : One split of the Hugging Face dataset (e.g. dataset['train'])
            tokenizer  : The DistilBERT tokeniser
            max_length : Maximum number of tokens per sentence (from config.py)
        """
        self.data      = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the total number of examples in this split."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single tokenised example.

        Args:
            idx : Integer index of the example to retrieve

        Returns:
            A dict with 'input_ids', 'attention_mask', and 'label' tensors.
        """
        example  = self.data[idx]
        sentence = example['sentence']
        label    = example['label']

        # tokenizer() does several things at once:
        #   - Splits sentence into tokens
        #   - Converts tokens to integer IDs
        #   - Adds [CLS] at start and [SEP] at end
        #   - Pads short sentences to max_length with zeros
        #   - Truncates long sentences to max_length
        #   - Creates attention_mask (1 for real, 0 for padding)
        encoding = self.tokenizer(
            sentence,
            max_length      = self.max_length,
            padding         = 'max_length',   # pad to exactly max_length
            truncation      = True,           # cut if longer than max_length
            return_tensors  = 'pt',           # return PyTorch tensors (not lists)
        )

        # encoding returns tensors of shape [1, max_length] — squeeze removes the
        # extra dimension so we get shape [max_length] instead
        return {
            'input_ids'      : encoding['input_ids'].squeeze(0),       # [128]
            'attention_mask' : encoding['attention_mask'].squeeze(0),  # [128]
            'label'          : torch.tensor(label, dtype=torch.long),  # scalar
        }


# =============================================================================
# STEP 5 — CREATE DATALOADERS
# =============================================================================
# WHAT IS A DATALOADER?
#
# Training on one example at a time is very slow. A DataLoader groups examples
# into "batches" (e.g. 16 examples at once) and feeds them to the GPU together.
# This is called "mini-batch gradient descent" and is how all modern models train.
#
# Key DataLoader arguments we use:
#   • batch_size  : Number of examples per batch (16 from config.py)
#   • shuffle     : If True, randomise the order each epoch (only for training!)
#   • num_workers : Number of parallel CPU threads for data loading.
#                   We use 0 on Mac because macOS has issues with multiprocessing
#                   in PyTorch's DataLoader with num_workers > 0.
# =============================================================================

def create_dataloaders(dataset, tokenizer):
    """
    Wrap the dataset splits into PyTorch DataLoaders ready for training.

    Args:
        dataset   : The Hugging Face dataset dict (from load_sst2_dataset)
        tokenizer : The loaded DistilBERT tokeniser

    Returns:
        train_loader : DataLoader for training (shuffled)
        val_loader   : DataLoader for validation (not shuffled)
        test_loader  : DataLoader for test (not shuffled, labels unavailable)
    """
    print(f"\n[DataLoader] Creating datasets...")

    # Wrap each split in our custom Dataset class
    train_dataset = SST2Dataset(dataset['train'],      tokenizer)
    val_dataset   = SST2Dataset(dataset['validation'], tokenizer)
    test_dataset  = SST2Dataset(dataset['test'],       tokenizer)

    print(f"[DataLoader] Train    : {len(train_dataset):,} examples")
    print(f"[DataLoader] Validate : {len(val_dataset):,} examples")
    print(f"[DataLoader] Test     : {len(test_dataset):,} examples")

    # num_workers=0 is required on macOS to avoid multiprocessing errors
    # persistent_workers=False is consistent with num_workers=0
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,           # Shuffle training data every epoch
        num_workers = 0,              # No parallel workers (macOS compatibility)
        pin_memory  = False,          # pin_memory=True is only useful for CUDA, not MPS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,          # Never shuffle evaluation data
        num_workers = 0,
        pin_memory  = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    # Calculate and display total number of batches
    print(f"\n[DataLoader] Batch size    : {BATCH_SIZE}")
    print(f"[DataLoader] Train batches : {len(train_loader):,}")
    print(f"[DataLoader] Val batches   : {len(val_loader):,}")
    print(f"[DataLoader] Test batches  : {len(test_loader):,}")

    return train_loader, val_loader, test_loader


# =============================================================================
# STEP 6 — UTILITY: INSPECT A BATCH
# =============================================================================
# This is a debugging helper. It takes one batch from a DataLoader and
# prints the shapes and values so you can verify everything looks right
# before starting the long training process.
# =============================================================================

def inspect_batch(loader, tokenizer, split_name: str = "train"):
    """
    Pull one batch from a DataLoader and print its contents for inspection.

    Args:
        loader     : A PyTorch DataLoader
        tokenizer  : The tokeniser (used to decode IDs back to text)
        split_name : Name of the split (for display purposes)
    """
    print(f"\n{'='*60}")
    print(f"  BATCH INSPECTION — {split_name.upper()} SPLIT")
    print(f"{'='*60}")

    # iter() + next() pulls the very first batch without looping through all data
    batch = next(iter(loader))

    input_ids      = batch['input_ids']       # Shape: [batch_size, max_seq_len]
    attention_mask = batch['attention_mask']  # Shape: [batch_size, max_seq_len]
    labels         = batch['label']           # Shape: [batch_size]

    print(f"\nTensor shapes:")
    print(f"  input_ids shape      : {list(input_ids.shape)}")
    print(f"             meaning   : [batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]}]")
    print(f"  attention_mask shape : {list(attention_mask.shape)}")
    print(f"  labels shape         : {list(labels.shape)}")

    print(f"\nLabel distribution in this batch:")
    n_positive = (labels == 1).sum().item()
    n_negative = (labels == 0).sum().item()
    print(f"  Positive (1): {n_positive}  |  Negative (0): {n_negative}")

    # Decode the first 2 examples back to readable text
    print(f"\nDecoded examples (first 2 in batch):")
    for i in range(min(2, BATCH_SIZE)):
        # Convert token IDs back to text, skipping special tokens like [CLS], [SEP], [PAD]
        text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        label_word = "POSITIVE" if labels[i].item() == 1 else "NEGATIVE"
        real_tokens = attention_mask[i].sum().item()  # Count non-padding tokens
        print(f"\n  Example {i}:")
        print(f"    Label      : {label_word}")
        print(f"    Real tokens: {int(real_tokens)} / {MAX_SEQ_LENGTH} (rest are padding)")
        print(f"    Text       : \"{text[:100]}\"")

    print(f"\n{'='*60}")


# =============================================================================
# MAIN FUNCTION — Runs when you execute: python data_pipeline.py
# =============================================================================
# This section only runs if you execute this file directly.
# When train.py imports from data_pipeline.py, this block is SKIPPED.
# That's what `if __name__ == "__main__":` means.
# =============================================================================

def get_data_pipeline():
    """
    Master function that runs the full data pipeline and returns everything
    needed for training.

    Returns:
        train_loader : DataLoader for training
        val_loader   : DataLoader for validation
        test_loader  : DataLoader for test
        tokenizer    : The loaded tokeniser (also needed in model.py)
    """
    set_seed(SEED)
    dataset   = load_sst2_dataset()
    tokenizer = load_tokenizer()
    train_loader, val_loader, test_loader = create_dataloaders(dataset, tokenizer)
    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Run a full pipeline test.
    # This verifies everything works before you spend hours training a model.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  R2-FORMER v2 — DATA PIPELINE TEST")
    print("=" * 60)

    # Step 1: Set seed
    set_seed(SEED)

    # Step 2: Load SST-2 dataset
    dataset = load_sst2_dataset()

    # Step 3: Load tokeniser
    tokenizer = load_tokenizer()

    # Step 4: Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset, tokenizer)

    # Step 5: Inspect a batch from training data
    inspect_batch(train_loader, tokenizer, split_name="train")

    # Step 6: Inspect a batch from validation data
    inspect_batch(val_loader, tokenizer, split_name="validation")

    # Step 7: Quick tokenisation demo — show raw token IDs for one sentence
    print("\n[Demo] Tokenisation example:")
    demo_sentence = "The acting was surprisingly good, but the plot fell flat."
    tokens = tokenizer.tokenize(demo_sentence)
    ids    = tokenizer.convert_tokens_to_ids(tokens)
    print(f"  Sentence : \"{demo_sentence}\"")
    print(f"  Tokens   : {tokens}")
    print(f"  IDs      : {ids}")

    print("\n[✓] Data pipeline test complete. All checks passed!")
    print("[✓] You are ready to move on to model.py")
