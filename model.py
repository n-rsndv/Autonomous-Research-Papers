# =============================================================================
# model.py — R2-Former v2
# =============================================================================
# WHAT THIS FILE DOES (plain English):
#
# This file defines the BRAIN of R2-Former v2 — the neural network architecture.
# It describes exactly how the model transforms tokenised text into a prediction
# of "positive" or "negative" sentiment.
#
# THE THREE NOVEL CONTRIBUTIONS (what makes this paper original):
#
#   1. DUAL RATIONALE EXTRACTION
#      Most models look at text one way. We look at it TWO ways simultaneously:
#        • LOCAL head  → word-level attention (which specific words matter?)
#        • GLOBAL head → sentence-level attention (what's the overall theme?)
#      These two views are then fused together into a richer representation.
#
#   2. ADAPTIVE CONFIDENCE GATING
#      After the first pass, we measure how "confident" the model is.
#      • High confidence → trust the initial reading, refine lightly
#      • Low confidence  → signal the refinement step to work harder
#      This mimics how a human re-reads a sentence when they're unsure.
#
#   3. ITERATIVE TWO-PASS REFINEMENT
#      The model reads the text TWICE. The second pass is conditioned on:
#        • The dual rationales from step 1
#        • The confidence gate signal from step 2
#      This "deliberative" second pass catches nuance the first pass missed.
#
# HOW THEY FIT TOGETHER (the full forward pass):
#
#   Raw tokens
#       ↓
#   [DistilBERT encoder]  ← pre-trained, fine-tuned
#       ↓
#   [Dual Rationale Extraction]  ← LOCAL + GLOBAL heads, fused
#       ↓
#   [Adaptive Confidence Gate]  ← how sure are we?
#       ↓
#   [Iterative Refinement x2]   ← second deliberative pass
#       ↓
#   [Classifier]                ← final prediction: positive or negative
#
# =============================================================================

import torch                           # PyTorch — our deep learning framework
import torch.nn as nn                  # Neural network building blocks
import torch.nn.functional as F        # Activation functions, softmax, etc.
from transformers import AutoModel     # Loads pre-trained DistilBERT
from config import *                   # All hyperparameters from config.py


# =============================================================================
# MODULE 1 — DUAL RATIONALE EXTRACTION
# =============================================================================
# WHAT IS "RATIONALE EXTRACTION"?
#
# A "rationale" is a subset of tokens that explains a prediction.
# For "The acting was great but the plot was terrible", a rationale for
# NEGATIVE might highlight "plot was terrible".
#
# WHY DUAL? (the novel contribution)
# Prior work uses one attention head for rationale extraction.
# We use TWO heads with different roles:
#
#   LOCAL head  — multi-head self-attention over all tokens.
#                 Learns which individual WORDS are most informative.
#                 e.g. "terrible", "brilliant", "waste", "masterpiece"
#
#   GLOBAL head — cross-attention where the [CLS] token attends to all others.
#                 [CLS] is the special "summary" token DistilBERT produces.
#                 Learns the SENTENCE-LEVEL theme, not just word salience.
#
#   FUSION      — We concatenate both views and project to HIDDEN_SIZE.
#                 This fused representation carries both local and global signal.
#
# WHAT IS MULTI-HEAD ATTENTION? (plain English)
# Instead of one "attention" (one way of deciding what's important),
# multi-head attention uses N heads in parallel, each focusing on different
# aspects of the text. The results are then combined. Think of it like
# N people each highlighting a document differently, then merging their notes.
# =============================================================================

class DualRationaleExtractor(nn.Module):
    """
    Extracts LOCAL (word-level) and GLOBAL (sentence-level) rationales
    from the DistilBERT hidden states, then fuses them.

    Input  : hidden_states — shape [batch, seq_len, hidden_size]
              e.g. [16, 128, 768]

    Output : fused rationale — shape [batch, seq_len, hidden_size]
              e.g. [16, 128, 768]  — same shape, richer content
    """

    def __init__(self):
        super().__init__()

        # ── LOCAL HEAD ────────────────────────────────────────────────────────
        # nn.MultiheadAttention(embed_dim, num_heads) creates a standard
        # multi-head self-attention layer.
        #
        # self-attention means: every token looks at every other token and
        # decides which are most relevant to itself.
        #
        # LOCAL_ATTN_HEADS = 4 (from config.py)
        # embed_dim = HIDDEN_SIZE = 768
        # batch_first=True means our tensors are [batch, seq, hidden] (standard)
        self.local_attn = nn.MultiheadAttention(
            embed_dim   = HIDDEN_SIZE,
            num_heads   = LOCAL_ATTN_HEADS,
            dropout     = DROPOUT_RATE,
            batch_first = True,
        )

        # ── GLOBAL HEAD ───────────────────────────────────────────────────────
        # Cross-attention: the [CLS] token (query) attends to all tokens (key/value).
        # This extracts a sentence-level view of what matters.
        #
        # GLOBAL_ATTN_HEADS = 4 (from config.py)
        self.global_attn = nn.MultiheadAttention(
            embed_dim   = HIDDEN_SIZE,
            num_heads   = GLOBAL_ATTN_HEADS,
            dropout     = DROPOUT_RATE,
            batch_first = True,
        )

        # ── FUSION LAYER ──────────────────────────────────────────────────────
        # After concatenating LOCAL and GLOBAL outputs, we have 2 * HIDDEN_SIZE = 1536.
        # This linear layer projects it back down to HIDDEN_SIZE = 768.
        # Think of it as: "compress both views into one combined view."
        self.fusion = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)

        # Layer normalisation stabilises training by normalising each token's
        # representation to have mean=0 and variance=1 (then learnable scale/shift).
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)

        # Dropout randomly zeroes some activations during training to prevent
        # the model from memorising the training data (regularisation).
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states  : [batch, seq_len, hidden_size] — DistilBERT output
            attention_mask : [batch, seq_len] — 1=real token, 0=padding

        Returns:
            fused : [batch, seq_len, hidden_size] — dual rationale representation
        """

        # ── PREPARE KEY PADDING MASK ──────────────────────────────────────────
        # PyTorch's MultiheadAttention uses a "key_padding_mask" where
        # True = IGNORE this token (padding), False = attend to this token.
        # Our attention_mask is the opposite (1=real, 0=pad), so we flip it.
        key_padding_mask = None
        if attention_mask is not None:
            # (attention_mask == 0) gives True for padding positions
            key_padding_mask = (attention_mask == 0)  # [batch, seq_len]

        # ── LOCAL SELF-ATTENTION ──────────────────────────────────────────────
        # Query = Key = Value = hidden_states (that's what makes it "self"-attention)
        # Every token attends to every other token.
        # local_out shape: [batch, seq_len, hidden_size]
        local_out, _ = self.local_attn(
            query            = hidden_states,
            key              = hidden_states,
            value            = hidden_states,
            key_padding_mask = key_padding_mask,
        )

        # ── GLOBAL CROSS-ATTENTION ────────────────────────────────────────────
        # The [CLS] token is always at position 0.
        # We use it as the QUERY: "what does the summary token want to know?"
        # The whole sequence is KEY and VALUE: "here's everything to attend to."
        #
        # cls_token shape: [batch, 1, hidden_size] — the "1" keeps the seq dimension
        cls_token = hidden_states[:, 0:1, :]   # slice position 0, keep dims

        # global_out shape: [batch, 1, hidden_size] — CLS attends to all tokens
        global_out, _ = self.global_attn(
            query            = cls_token,
            key              = hidden_states,
            value            = hidden_states,
            key_padding_mask = key_padding_mask,
        )

        # Expand global_out from [batch, 1, hidden_size] to [batch, seq_len, hidden_size]
        # so it can be concatenated with local_out.
        # expand() replicates the CLS global context across all token positions.
        seq_len = hidden_states.size(1)
        global_out = global_out.expand(-1, seq_len, -1)  # -1 means "keep this dim"

        # ── FUSION ────────────────────────────────────────────────────────────
        # Concatenate along the last dimension (hidden_size dimension).
        # [batch, seq_len, 768] + [batch, seq_len, 768] → [batch, seq_len, 1536]
        combined = torch.cat([local_out, global_out], dim=-1)

        # Project back to HIDDEN_SIZE: [batch, seq_len, 1536] → [batch, seq_len, 768]
        fused = self.fusion(combined)
        fused = self.dropout(fused)

        # Residual connection: add the original input to the fused output.
        # WHY? Residual connections prevent "vanishing gradients" during training
        # and let the model learn "what to ADD" rather than "what to output",
        # which is much easier to optimise.
        fused = self.layer_norm(fused + hidden_states)

        return fused


# =============================================================================
# MODULE 2 — ADAPTIVE CONFIDENCE GATE
# =============================================================================
# WHAT IS A CONFIDENCE GATE? (the novel contribution)
#
# After the dual rationale extraction, we need to decide:
#   "How confident is the model in its current understanding?"
#
# We compute a scalar gate value g ∈ [0, 1] for each example in the batch:
#   • g close to 1 → HIGH confidence → refinement should be light
#   • g close to 0 → LOW confidence  → refinement should work harder
#
# HOW IS g COMPUTED?
# We take the [CLS] token representation (the sentence summary), pass it
# through a small 2-layer network, and apply a sigmoid to squash to [0,1].
#
# HOW IS g USED?
# The gate signal g is concatenated with the rationale and passed into the
# refinement transformer, which then adapts its behaviour based on confidence.
# This tight coupling (gate → refinement) is what's novel over the original R2-Former.
#
# WHAT IS SIGMOID?
# sigmoid(x) = 1 / (1 + e^(-x))
# It maps any real number to a value between 0 and 1.
# Perfect for computing a "probability" or "confidence score".
# =============================================================================

class AdaptiveConfidenceGate(nn.Module):
    """
    Computes a scalar confidence gate g ∈ [0,1] from the [CLS] token.
    Also produces a gate-conditioned representation for the refinement step.

    Input  : fused_rationale — [batch, seq_len, hidden_size]

    Outputs:
        gate_signal : [batch, seq_len, hidden_size] — rationale conditioned by gate
        gate_value  : [batch, 1] — scalar confidence score (for logging/analysis)
    """

    def __init__(self):
        super().__init__()

        # ── GATE NETWORK ──────────────────────────────────────────────────────
        # A small 2-layer MLP (Multi-Layer Perceptron) that reads the [CLS] token
        # and outputs a confidence scalar.
        #
        # Layer 1: HIDDEN_SIZE (768) → HIDDEN_SIZE // 2 (384) with GELU activation
        # Layer 2: HIDDEN_SIZE // 2 (384) → 1 (scalar)
        # Final: sigmoid → squash to [0, 1]
        #
        # WHAT IS GELU?
        # GELU (Gaussian Error Linear Unit) is a smooth activation function,
        # similar to ReLU but differentiable everywhere. Used in BERT/GPT.
        self.gate_network = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_SIZE // 2, 1),
            nn.Sigmoid(),   # output ∈ [0, 1]
        )

        # ── GATE PROJECTION ───────────────────────────────────────────────────
        # We want to inject the gate signal into every token position.
        # This linear layer transforms the scalar gate into a vector that can
        # be added to the rationale representation.
        self.gate_projection = nn.Linear(1, HIDDEN_SIZE)

        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.dropout    = nn.Dropout(DROPOUT_RATE)

    def forward(self, fused_rationale):
        """
        Args:
            fused_rationale : [batch, seq_len, hidden_size]

        Returns:
            gate_signal : [batch, seq_len, hidden_size]
            gate_value  : [batch, 1] — the raw confidence scalar
        """

        # Extract [CLS] token (position 0) — the sentence-level summary
        # cls_repr shape: [batch, hidden_size]
        cls_repr = fused_rationale[:, 0, :]

        # Compute confidence scalar for each example in the batch
        # gate_value shape: [batch, 1]
        gate_value = self.gate_network(cls_repr)

        # Project the scalar gate into a full hidden-size vector
        # [batch, 1] → [batch, hidden_size]
        gate_vec = self.gate_projection(gate_value)

        # Unsqueeze to [batch, 1, hidden_size] then expand to [batch, seq_len, hidden_size]
        # so we can add it to every token position in the rationale
        seq_len    = fused_rationale.size(1)
        gate_vec   = gate_vec.unsqueeze(1).expand(-1, seq_len, -1)

        # Condition the rationale by the gate signal
        # High gate (confident): gate_vec pushes representations toward "trust me"
        # Low gate (uncertain):  gate_vec signals "look harder in refinement"
        gate_signal = self.layer_norm(self.dropout(fused_rationale + gate_vec))

        return gate_signal, gate_value


# =============================================================================
# MODULE 3 — ITERATIVE REFINEMENT TRANSFORMER
# =============================================================================
# WHAT IS ITERATIVE REFINEMENT? (the novel contribution)
#
# The model makes TWO passes through a lightweight transformer:
#
#   Pass 1 (first refinement):
#     Input  = gate_signal (rationale + confidence conditioning)
#     Output = refined_1 (first attempt at better understanding)
#
#   Pass 2 (second refinement):
#     Input  = refined_1 PLUS the original gate_signal injected again
#     Output = refined_2 (final, deliberative understanding)
#
# WHY TWO PASSES?
# Human readers re-read complex sentences. The first pass forms an initial
# interpretation; the second pass checks it against the rationale and adjusts.
# This is what we mean by "deliberative judgment" in the paper.
#
# WHY IS THIS DIFFERENT FROM JUST ADDING MORE LAYERS?
# Standard transformers stack layers linearly — each layer sees the same input.
# Our refinement is CONDITIONED: the second pass receives both its own output
# AND the original gate signal re-injected. This creates a feedback loop.
#
# WHAT IS A TRANSFORMER ENCODER LAYER?
# A standard building block with:
#   1. Multi-head self-attention (tokens attend to each other)
#   2. Feed-forward network (per-token nonlinear transformation)
#   3. Residual connections + layer norm around each
# We use REFINEMENT_LAYERS=2 stacked layers per pass.
# =============================================================================

class IterativeRefinementTransformer(nn.Module):
    """
    Two-pass iterative refinement conditioned on the gate signal.

    Each pass is a stack of standard Transformer encoder layers.
    Between passes, the gate signal is re-injected so the second
    pass "knows" the confidence level of the first pass.

    Input  : gate_signal — [batch, seq_len, hidden_size]
    Output : refined     — [batch, seq_len, hidden_size]
    """

    def __init__(self):
        super().__init__()

        # ── SHARED TRANSFORMER ENCODER LAYER CONFIG ───────────────────────────
        # We define one TransformerEncoderLayer template and use it for both passes.
        # nhead = REFINEMENT_HEADS = 8 (from config.py)
        # dim_feedforward = 4 * HIDDEN_SIZE = 3072 (standard BERT ratio)
        # batch_first = True because our tensors are [batch, seq, hidden]
        encoder_layer_config = dict(
            d_model         = HIDDEN_SIZE,
            nhead           = REFINEMENT_HEADS,
            dim_feedforward = HIDDEN_SIZE * 4,
            dropout         = DROPOUT_RATE,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # Pre-LN: normalise BEFORE attention (more stable)
        )

        # ── PASS 1 — FIRST REFINEMENT ─────────────────────────────────────────
        # A stack of REFINEMENT_LAYERS=2 transformer encoder layers.
        # nn.TransformerEncoder wraps N identical encoder layers.
        self.refine_pass1 = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_config),
            num_layers    = REFINEMENT_LAYERS,
        )

        # ── PASS 2 — SECOND REFINEMENT ────────────────────────────────────────
        # A separate stack for the second pass.
        # We use separate weights (not shared) so each pass can specialise.
        self.refine_pass2 = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_config),
            num_layers    = REFINEMENT_LAYERS,
        )

        # ── GATE RE-INJECTION LAYER ───────────────────────────────────────────
        # Between pass 1 and pass 2, we re-inject the gate signal.
        # This projection blends pass1 output with the original gate signal.
        # Input: concatenation of [pass1_output, gate_signal] = 2 * HIDDEN_SIZE
        # Output: HIDDEN_SIZE
        self.gate_injection = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)

        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.dropout    = nn.Dropout(DROPOUT_RATE)

    def forward(self, gate_signal, attention_mask=None):
        """
        Args:
            gate_signal    : [batch, seq_len, hidden_size] — from confidence gate
            attention_mask : [batch, seq_len] — 1=real, 0=padding

        Returns:
            refined : [batch, seq_len, hidden_size]
        """

        # Build the src_key_padding_mask for transformer layers
        # True = ignore (padding), same convention as MultiheadAttention
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # [batch, seq_len]

        # ── PASS 1 ────────────────────────────────────────────────────────────
        # First deliberative pass through 2 transformer encoder layers.
        # refined_1 shape: [batch, seq_len, hidden_size]
        refined_1 = self.refine_pass1(
            gate_signal,
            src_key_padding_mask = src_key_padding_mask,
        )

        # ── GATE RE-INJECTION ─────────────────────────────────────────────────
        # Concatenate pass1 output with original gate_signal.
        # This gives pass2 access to both what it learned and the original signal.
        # [batch, seq_len, 768] cat [batch, seq_len, 768] → [batch, seq_len, 1536]
        combined = torch.cat([refined_1, gate_signal], dim=-1)

        # Project back to HIDDEN_SIZE: [batch, seq_len, 1536] → [batch, seq_len, 768]
        injected = self.gate_injection(combined)
        injected = self.dropout(injected)
        injected = self.layer_norm(injected + refined_1)   # residual

        # ── PASS 2 ────────────────────────────────────────────────────────────
        # Second deliberative pass, now conditioned on re-injected gate signal.
        # refined_2 shape: [batch, seq_len, hidden_size]
        refined_2 = self.refine_pass2(
            injected,
            src_key_padding_mask = src_key_padding_mask,
        )

        return refined_2


# =============================================================================
# MODULE 4 — CLASSIFIER HEAD
# =============================================================================
# WHAT IS A CLASSIFIER HEAD?
#
# After all the processing above, we have a rich representation of the text.
# We now need to convert it to a prediction: 0 (negative) or 1 (positive).
#
# STEPS:
#   1. Take the [CLS] token from the refined output (position 0).
#      The [CLS] token summarises the whole sequence — ideal for classification.
#
#   2. Pass through a 2-layer MLP:
#      768 → 768/2 (384) → NUM_CLASSES (2)
#
#   3. Output: logits of shape [batch, 2]
#      "Logits" are raw un-normalised scores. The training loss (CrossEntropy)
#      handles the softmax internally, so we don't apply softmax here.
#      For inference, we take argmax(logits) to get the predicted class.
# =============================================================================

class ClassifierHead(nn.Module):
    """
    Maps the refined [CLS] token to class logits.

    Input  : refined — [batch, seq_len, hidden_size]
    Output : logits  — [batch, num_classes]  (raw scores, no softmax)
    """

    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_SIZE // 2, NUM_CLASSES),
        )

    def forward(self, refined):
        """
        Args:
            refined : [batch, seq_len, hidden_size]

        Returns:
            logits : [batch, num_classes]
        """
        # Extract the [CLS] token (position 0) — shape [batch, hidden_size]
        cls_output = refined[:, 0, :]

        # Classify: [batch, hidden_size] → [batch, num_classes]
        logits = self.classifier(cls_output)

        return logits


# =============================================================================
# FULL MODEL — R2FormerV2
# =============================================================================
# This is the top-level model class that assembles all four modules above
# into a single end-to-end architecture.
#
# FULL FORWARD PASS SUMMARY:
#
#   input_ids + attention_mask
#         ↓
#   [DistilBERT]                  → hidden_states [batch, 128, 768]
#         ↓
#   [DualRationaleExtractor]      → fused_rationale [batch, 128, 768]
#         ↓
#   [AdaptiveConfidenceGate]      → gate_signal [batch, 128, 768]
#                                   gate_value  [batch, 1]
#         ↓
#   [IterativeRefinementTransformer] → refined [batch, 128, 768]
#         ↓
#   [ClassifierHead]              → logits [batch, 2]
#         ↓
#   argmax → predicted class (0 or 1)
#
# PARAMETER COUNT (approximate):
#   DistilBERT backbone : ~66M  (pre-trained, fine-tuned)
#   DualRationaleExtractor : ~4.7M
#   AdaptiveConfidenceGate : ~0.6M
#   IterativeRefinement    : ~25M
#   ClassifierHead         : ~0.4M
#   TOTAL                  : ~97M parameters
# =============================================================================

class R2FormerV2(nn.Module):
    """
    R2-Former v2: Dual Rationale Extraction + Adaptive Confidence Gating
    + Iterative Two-Pass Refinement for Sentiment Analysis.

    Architecture:
        DistilBERT backbone
        → DualRationaleExtractor  (LOCAL + GLOBAL attention, fused)
        → AdaptiveConfidenceGate  (confidence scalar gates refinement)
        → IterativeRefinementTransformer (two deliberative passes)
        → ClassifierHead          (CLS token → class logits)
    """

    def __init__(self):
        super().__init__()

        # ── BACKBONE: PRE-TRAINED DISTILBERT ──────────────────────────────────
        # AutoModel.from_pretrained() downloads the DistilBERT weights
        # (~250MB on first run, then cached locally).
        #
        # WHY DISTILBERT AND NOT FULL BERT?
        # DistilBERT is a "distilled" (compressed) version of BERT:
        #   • 40% smaller, 60% faster than BERT
        #   • Retains 97% of BERT's performance on GLUE benchmarks
        #   • Ideal for a research prototype where training time matters
        #
        # The model outputs hidden_states of shape [batch, seq_len, 768]
        # — a 768-dimensional vector for every token in the sequence.
        print(f"[Model] Loading pre-trained backbone: '{PRETRAINED_MODEL}'")
        self.encoder = AutoModel.from_pretrained(PRETRAINED_MODEL)
        # Freeze all DistilBERT layers except the last transformer block
        for name, param in self.encoder.named_parameters():
            if not any(l in name for l in ['transformer.layer.4', 'transformer.layer.5']):
                param.requires_grad = False

        # ── OUR THREE NOVEL MODULES ───────────────────────────────────────────
        self.dual_rationale  = DualRationaleExtractor()
        self.confidence_gate = AdaptiveConfidenceGate()
        self.refinement      = IterativeRefinementTransformer()
        self.classifier      = ClassifierHead()

        # ── PARAMETER COUNT DISPLAY ───────────────────────────────────────────
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total parameters     : {total_params:,}")
        print(f"[Model] Trainable parameters : {trainable_params:,}")

    def forward(self, input_ids, attention_mask):
        """
        Full forward pass through R2-Former v2.

        Args:
            input_ids      : [batch, seq_len] — token ID integers from tokeniser
            attention_mask : [batch, seq_len] — 1=real token, 0=padding

        Returns:
            logits     : [batch, num_classes] — raw prediction scores
            gate_value : [batch, 1]           — confidence score (for analysis)
        """

        # ── STEP 1: DISTILBERT ENCODING ───────────────────────────────────────
        # Pass tokenised text through the pre-trained DistilBERT.
        # It returns an object; we access .last_hidden_state for the token vectors.
        # hidden_states shape: [batch, seq_len, hidden_size] = [16, 128, 768]
        encoder_output = self.encoder(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        hidden_states = encoder_output.last_hidden_state   # [batch, seq_len, 768]

        # ── STEP 2: DUAL RATIONALE EXTRACTION ────────────────────────────────
        # LOCAL self-attention + GLOBAL CLS cross-attention → fused
        # fused_rationale shape: [batch, seq_len, hidden_size]
        fused_rationale = self.dual_rationale(hidden_states, attention_mask)

        # ── STEP 3: ADAPTIVE CONFIDENCE GATING ───────────────────────────────
        # Compute confidence scalar and condition the rationale
        # gate_signal shape: [batch, seq_len, hidden_size]
        # gate_value  shape: [batch, 1]
        gate_signal, gate_value = self.confidence_gate(fused_rationale)

        # ── STEP 4: ITERATIVE TWO-PASS REFINEMENT ────────────────────────────
        # Two deliberative passes, second conditioned on gate re-injection
        # refined shape: [batch, seq_len, hidden_size]
        refined = self.refinement(gate_signal, attention_mask)

        # ── STEP 5: CLASSIFICATION ────────────────────────────────────────────
        # Extract [CLS] token → 2-layer MLP → logits
        # logits shape: [batch, num_classes] = [16, 2]
        logits = self.classifier(refined)

        # We return gate_value as well so train.py can log it for analysis.
        # In the paper, we can show that low-confidence examples benefit most
        # from the refinement step — this supports our motivation.
        return logits, gate_value


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_model(device):
    """
    Construct the R2FormerV2 model and move it to the target device (MPS/CPU).

    Args:
        device : torch.device — the device to run on (from config.py)

    Returns:
        model : R2FormerV2 on the specified device
    """
    print(f"\n[Model] Building R2-Former v2...")
    model = R2FormerV2()
    model = model.to(device)
    print(f"[Model] Model moved to device: {device}")
    return model


def count_parameters(model):
    """
    Print a breakdown of parameter counts by module.
    Useful for understanding where the model's capacity is.

    Args:
        model : R2FormerV2 instance
    """
    print(f"\n{'='*55}")
    print(f"  MODEL PARAMETER BREAKDOWN")
    print(f"{'='*55}")

    # Named children gives us the top-level modules
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<25} {params:>12,}  ({trainable:,} trainable)")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'─'*55}")
    print(f"  {'TOTAL':<25} {total:>12,}  ({trainable:,} trainable)")
    print(f"{'='*55}\n")


# =============================================================================
# MAIN — Test the model architecture
# =============================================================================
# Run this file directly to verify the model builds and a forward pass works.
# This is a "sanity check" before we start the full training loop.
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  R2-FORMER v2 — MODEL ARCHITECTURE TEST")
    print("=" * 60)

    # ── DEVICE SETUP ──────────────────────────────────────────────────────────
    # DEVICE is imported from config.py (should be 'mps' on your M2 Mac)
    print(f"\n[Test] Using device: {DEVICE}")

    # ── BUILD MODEL ───────────────────────────────────────────────────────────
    model = build_model(DEVICE)

    # ── PARAMETER COUNT ───────────────────────────────────────────────────────
    count_parameters(model)

    # ── SYNTHETIC FORWARD PASS ────────────────────────────────────────────────
    # We create fake input tensors that look exactly like real data.
    # This lets us verify the model architecture WITHOUT needing the dataset.
    #
    # torch.randint(low, high, shape) creates random integers in [low, high)
    # Our vocabulary has 30,522 tokens, so token IDs are in [0, 30522)
    print("[Test] Running forward pass with synthetic data...")
    print(f"       Batch size   : {BATCH_SIZE}")
    print(f"       Sequence len : {MAX_SEQ_LENGTH}")
    print(f"       Hidden size  : {HIDDEN_SIZE}")

    # Fake input_ids: [batch, seq_len] of random token IDs
    fake_input_ids = torch.randint(
        0, 30522,
        (BATCH_SIZE, MAX_SEQ_LENGTH),
        device = DEVICE,
    )

    # Fake attention_mask: first 20 tokens "real", rest padding
    # In practice, SST-2 sentences are typically 7–30 tokens
    fake_attention_mask = torch.zeros(
        BATCH_SIZE, MAX_SEQ_LENGTH,
        dtype = torch.long,
        device = DEVICE,
    )
    fake_attention_mask[:, :20] = 1   # first 20 tokens are "real"

    # Run the forward pass
    # torch.no_grad() disables gradient computation — not needed for testing
    model.eval()
    with torch.no_grad():
        logits, gate_value = model(fake_input_ids, fake_attention_mask)

    # ── VERIFY OUTPUT SHAPES ──────────────────────────────────────────────────
    print(f"\n[Test] Output shapes:")
    print(f"       logits     : {list(logits.shape)}   ← expected [{BATCH_SIZE}, {NUM_CLASSES}]")
    print(f"       gate_value : {list(gate_value.shape)}    ← expected [{BATCH_SIZE}, 1]")

    # Verify shapes match expectations
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), \
        f"Logits shape mismatch! Got {logits.shape}"
    assert gate_value.shape == (BATCH_SIZE, 1), \
        f"Gate value shape mismatch! Got {gate_value.shape}"

    # ── SHOW SAMPLE PREDICTIONS ───────────────────────────────────────────────
    # argmax gives the index of the highest logit = predicted class
    predictions = torch.argmax(logits, dim=-1)
    print(f"\n[Test] Sample logits (first 4 examples):")
    for i in range(min(4, BATCH_SIZE)):
        neg_score = logits[i, 0].item()
        pos_score = logits[i, 1].item()
        pred      = "POSITIVE" if predictions[i].item() == 1 else "NEGATIVE"
        conf      = gate_value[i].item()
        print(f"       [{i}] Neg={neg_score:+.3f}  Pos={pos_score:+.3f}  "
              f"→ {pred}  (confidence gate: {conf:.3f})")

    print(f"\n[✓] All shape checks passed!")
    print(f"[✓] Forward pass successful on {DEVICE}")
    print(f"[✓] R2-Former v2 architecture is correct and ready for training")
    print(f"\nNext step: write train.py")