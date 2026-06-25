"""
QR-GPTQ: A numerically stable variant of GPTQ using QR decomposition of the
activation matrix in place of Cholesky decomposition of the Gram product.

Implements and benchmarks three weight-only PTQ methods on LLaMA-3 1B and 8B:
  1. GPTQ        -- LDL(H^{-1}) error propagation (Algorithm 1, Chen et al. 2025)
  2. QR-GPTQ     -- Householder QR of augmented activations (this work)
  3. ParoQuant   -- Scaled pairwise rotation + RTN (Liang et al. 2026, Stage 1 only)

Evaluation metrics:
  - Per-layer output MSE:  ||X W - X Q||_F^2 / (n * r)
  - Babai error bound:     (1/4) * tr(D^{(i)})  averaged over output channels
  - WikiText-2 perplexity  (requires model in fp16 as reference)
  - Wall-clock time per layer

Usage:
    python qr_gptq_benchmark.py \
        --model meta-llama/Llama-3.2-1B \
        --bits 4 \
        --group-size 128 \
        --n-calib 128 \
        --seq-len 2048 \
        --methods gptq qr_gptq paroquant \
        --device cuda

Requirements:
    pip install torch transformers datasets
    # HuggingFace token with LLaMA access set via:
    huggingface-cli login
"""

import argparse
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------------

def absmax_scale(W: Tensor, bits: int, group_size: int) -> Tuple[Tensor, Tensor]:
    """
    Compute per-group AbsMax scales and zero-points for symmetric INT quantization.

    W: (c, r)  -- rows are input channels, columns are output channels
    Returns:
        scales: (c, r)
        zeros:  (c, r)  -- all zeros for symmetric quant
    """
    c, r = W.shape
    n_groups = (c + group_size - 1) // group_size
    W_pad = torch.zeros(n_groups * group_size, r, dtype=W.dtype, device=W.device)
    W_pad[:c] = W
    W_g = W_pad.view(n_groups, group_size, r)           # (G, g, r)

    maxv = (2 ** (bits - 1)) - 1
    scale = W_g.abs().amax(dim=1, keepdim=True) / maxv  # (G, 1, r)
    scale = scale.clamp(min=1e-8)

    # Expand back to (c, r)
    scale_full = scale.expand(n_groups, group_size, r).reshape(n_groups * group_size, r)[:c]
    zero_full = torch.zeros_like(scale_full)
    return scale_full, zero_full


def mse_scale(W: Tensor, bits: int, group_size: int) -> Tuple[Tensor, Tensor]:
    """
    MSE-optimal scale via grid search over shrinkage factors in [0.8, 1.0].
    More accurate than AbsMax, especially for heavy-tailed channels.
    """
    best_scale, _ = absmax_scale(W, bits, group_size)
    best_mse = torch.full((W.shape[1],), float('inf'), device=W.device, dtype=W.dtype)

    for alpha in torch.linspace(0.8, 1.0, 20, device=W.device):
        s = best_scale * alpha
        W_q = quantize_dequantize(W, s, bits)
        mse = (W - W_q).pow(2).mean(dim=0)
        improve = mse < best_mse
        best_scale[:, improve] = s[:, improve]
        best_mse[improve] = mse[improve]

    return best_scale, torch.zeros_like(best_scale)


def round_to_grid(x: Tensor, bits: int, clip: bool = True) -> Tensor:
    """Round to nearest integer in [-(2^{b-1}), 2^{b-1}-1] (symmetric INT{b})."""
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax - 1
    x_int = x.round()
    if clip:
        x_int = x_int.clamp(qmin, qmax)
    return x_int


def quantize_dequantize(W: Tensor, scale: Tensor, bits: int, clip: bool = True) -> Tensor:
    """RTN quantize-dequantize: W_q = round(W / scale) * scale."""
    W_int = round_to_grid(W / scale, bits, clip)
    return W_int * scale


# ---------------------------------------------------------------------------
# Babai error bound
# ---------------------------------------------------------------------------

def babai_bound(A: Tensor, S: Tensor) -> float:
    """
    Compute the average-over-channels Babai absolute error bound:
        (1/4) * mean_i { (T^{-1} s_i)^T D (T^{-1} s_i) }
      = (1/4) * mean_i { sum_j A[j,j]^2 * (s_i[j])^2 }

    where A is the upper-triangular QR/Cholesky basis, D[j,j] = A[j,j]^2,
    and T is the identity (no reordering beyond what is baked into A).

    A: (c, c)  upper triangular
    S: (c, r)  per-element scales
    Returns scalar bound.
    """
    d = A.diag().pow(2)           # (c,)
    # For each output channel i, bound = 0.25 * sum_j d_j * s_i[j]^2
    # S: (c, r) => s_i = S[:, i]
    per_channel = 0.25 * (d.unsqueeze(1) * S.pow(2)).sum(dim=0)  # (r,)
    return per_channel.mean().item()


# ---------------------------------------------------------------------------
# Method 1: GPTQ (Algorithm 1, Chen et al. 2025)
# ---------------------------------------------------------------------------

def gptq_quantize_layer(
    W: Tensor,
    X: Tensor,
    scale: Tensor,
    bits: int,
    perm: Optional[Tensor] = None,
    damp_factor: float = 0.01,
) -> Tuple[Tensor, Tensor, float, float]:
    """
    GPTQ: LDL decomposition of H^{-1} with front-to-back error propagation.

    W:      (c, r)  weight matrix (rows = input dim, cols = output dim)
    X:      (n, c)  calibration activations
    scale:  (c, r)  per-element scales (from AbsMax or MSE)
    perm:   (c,)    optional quantization order permutation (act-order)
    Returns:
        Q:       (c, r)  dequantized weights
        Z:       (c, r)  integer weights
        mse:     per-element output MSE
        bound:   Babai bound (requires A from Cholesky, computed here)
    """
    c, r = W.shape
    device = W.device
    dtype = W.dtype

    # --- permutation ---
    if perm is None:
        perm = torch.arange(c, device=device)
    inv_perm = torch.argsort(perm)

    W = W[perm].clone().to(torch.float32)
    S = scale[perm].clone().to(torch.float32)
    X = X.to(torch.float32)

    # --- Hessian and damping ---
    H = X.T @ X                                           # (c, c)
    lam = damp_factor / c * H.trace() / c
    H = H[perm][:, perm]
    H.diagonal().add_(lam)

    # --- LDL of H^{-1} ---
    # Use Cholesky of H then invert: H = L_c L_c^T, H^{-1} = L_c^{-T} L_c^{-1}
    # L from LDL(H^{-1}): lower unit triangular
    try:
        L_c = torch.linalg.cholesky(H)                   # (c, c) lower triangular
    except torch.linalg.LinAlgError:
        H.diagonal().add_(1e-3 * H.diagonal().mean())
        L_c = torch.linalg.cholesky(H)

    # H^{-1} = (L_c L_c^T)^{-1} = L_c^{-T} L_c^{-1}
    # We need the lower unit triangular L such that H^{-1} = L D L^T
    # Equivalently: the error propagation column of GPTQ is H^{-1}[:, j] / H^{-1}[j,j]
    # Direct computation: form H_inv and extract its lower-triangular structure
    H_inv = torch.cholesky_inverse(L_c)                  # (c, c)

    # For GPTQ, at step j the update is:
    #   W[j+1:, :] += (H_inv[j+1:, j] / H_inv[j, j]) * eps
    # which equals L[j+1:, j] * eps with unit-diagonal L from LDL.
    # Pre-extract the columns: L_col[j] = H_inv[j:, j] / H_inv[j, j]
    # Store full H_inv; column access is cheap.

    # Babai bound: A[j,j]^2 = 1 / H_inv[j,j]  (diagonal of Cholesky of H)
    A_diag = L_c.diagonal()                              # (c,)

    Q = W.clone()
    Z = torch.zeros_like(W)

    for j in range(c):
        w_j = Q[j]                                       # (r,)
        s_j = S[j]                                       # (r,)
        zeta = w_j / s_j
        z_j = round_to_grid(zeta, bits, clip=True)
        q_j = z_j * s_j
        eps = q_j - w_j                                  # quantization error (r,)
        Z[j] = z_j
        Q[j] = q_j
        if j + 1 < c:
            col = H_inv[j + 1:, j] / H_inv[j, j]        # (c-j-1,)
            Q[j + 1:] += col.unsqueeze(1) * eps.unsqueeze(0)

    # Restore permutation
    Q = Q[inv_perm]
    Z = Z[inv_perm]
    S_orig = S[inv_perm]

    # Build scale-adjusted A_diag for bound (scale already incorporated)
    A_full = torch.zeros(c, c, device=device, dtype=torch.float32)
    A_full[torch.arange(c), torch.arange(c)] = A_diag
    bound = babai_bound(A_full, S_orig)

    # Output MSE
    W_orig = W[inv_perm]
    mse = (X.to(torch.float32) @ (W_orig - Q)).pow(2).mean().item()

    return Q.to(dtype), Z.to(torch.int8), mse, bound


# ---------------------------------------------------------------------------
# Method 2: QR-GPTQ (this work)
# ---------------------------------------------------------------------------

def qr_gptq_quantize_layer(
    W: Tensor,
    X: Tensor,
    scale: Tensor,
    bits: int,
    perm: Optional[Tensor] = None,
    damp_factor: float = 0.01,
) -> Tuple[Tensor, Tensor, float, float]:
    """
    QR-GPTQ: Householder QR of the augmented activation matrix.

    Factorises X_aug = [X * T; sqrt(lam) * I] = Phi * A  (thin QR)
    where A is upper triangular and satisfies A^T A = H = T^T(X^T X + lam I)T.

    The Babai inner loop is then run on A directly (Algorithm 4, Chen et al. 2025),
    without ever forming the c x c Gram product X^T X.
    """
    c, r = W.shape
    n = X.shape[0]
    device = W.device
    dtype = W.dtype

    # --- permutation ---
    if perm is None:
        perm = torch.arange(c, device=device)
    inv_perm = torch.argsort(perm)

    W = W[perm].clone().to(torch.float32)
    S = scale[perm].clone().to(torch.float32)
    X_fp = X.to(torch.float32)[:, perm]                  # (n, c) reordered

    # --- damping: augment with sqrt(lam) * I ---
    lam_scalar = damp_factor / c * (X_fp.T @ X_fp).trace().item() / c
    sqrt_lam = math.sqrt(max(lam_scalar, 1e-8))
    X_aug = torch.cat(
        [X_fp, sqrt_lam * torch.eye(c, device=device, dtype=torch.float32)],
        dim=0
    )                                                    # (n+c, c)

    # --- thin QR: X_aug = Phi * A, A upper triangular (c, c) ---
    # torch.linalg.qr with mode='reduced' gives Phi:(n+c, c), A:(c, c)
    _, A = torch.linalg.qr(X_aug, mode='reduced')       # A: (c, c) upper triangular

    # --- Babai back-substitution (Algorithm 4 of Chen et al.) ---
    # Y = A W  (target in the A-basis)
    Y = A @ W                                            # (c, r)
    Q = W.clone()
    Z = torch.zeros_like(W)

    for j in range(c - 1, -1, -1):
        omega = Y[j]                                     # (r,)
        zeta = omega / (A[j, j] * S[j])                 # (r,)
        z_j = round_to_grid(zeta, bits, clip=True)
        q_j = z_j * S[j]
        Z[j] = z_j
        Q[j] = q_j
        Y -= A[:, j].unsqueeze(1) * q_j.unsqueeze(0)    # update residual

    # Restore permutation
    Q = Q[inv_perm]
    Z = Z[inv_perm]
    S_orig = S[inv_perm]

    # Babai bound: D[j,j] = A[j,j]^2
    bound = babai_bound(A, S_orig)

    # Output MSE
    W_orig = W[inv_perm]
    X_eval = X.to(torch.float32)
    mse = (X_eval @ (W_orig - Q)).pow(2).mean().item()

    return Q.to(dtype), Z.to(torch.int8), mse, bound


# ---------------------------------------------------------------------------
# Method 3: ParoQuant (Stage 1: optimise transform then RTN)
# ---------------------------------------------------------------------------

def select_independent_pairs(g: int, n_rotations: int, n_pairs: int) -> List[List[Tuple[int, int]]]:
    """
    Algorithm A1 from Liang et al. 2026.

    For each of n_rotations independent rotations, randomly select up to n_pairs
    channel pairs that are mutually non-overlapping (independent).
    Pairs used in previous rotations are excluded to improve diversity.
    """
    import random
    all_pairs = [(i, j) for i in range(g) for j in range(i + 1, g)]
    random.shuffle(all_pairs)

    used_globally = set()
    result = []
    for _ in range(n_rotations):
        used_local = set()
        rotation_pairs = []
        for (i, j) in all_pairs:
            if len(rotation_pairs) >= n_pairs:
                break
            if (i, j) in used_globally:
                continue
            if i in used_local or j in used_local:
                continue
            rotation_pairs.append((i, j))
            used_local.add(i)
            used_local.add(j)
            used_globally.add((i, j))
        result.append(rotation_pairs)
    return result


def apply_givens_to_weight(W: Tensor, pairs: List[Tuple[int, int]], thetas: Tensor) -> Tensor:
    """
    Apply a sequence of Givens rotations to rows of W (in-place on a clone).
    W: (c, r) or (g, r) for a group
    thetas: (len(pairs),)
    """
    W = W.clone()
    for k, (i, j) in enumerate(pairs):
        c_k = thetas[k].cos()
        s_k = thetas[k].sin()
        row_i = W[i].clone()
        row_j = W[j].clone()
        W[i] = c_k * row_i - s_k * row_j
        W[j] = s_k * row_i + c_k * row_j
    return W


def paroquant_quantize_layer(
    W: Tensor,
    X: Tensor,
    scale_init: Tensor,
    bits: int,
    group_size: int = 128,
    n_rotations: int = 8,
    n_steps: int = 200,
    lr: float = 0.05,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, float, float]:
    """
    ParoQuant Stage 1: optimise scaled pairwise rotation to minimise ||X Q(T(W)) - X W||.
    Stage 2 (QAT fine-tuning) is omitted; the paper shows Stage 1 alone gives most of the gain.

    W:          (c, r)
    X:          (n, c)
    scale_init: (c, r)  initial scales (MSE or AbsMax)
    Returns Q, Z, mse, bound=nan (no closed-form bound for ParoQuant)
    """
    if device is None:
        device = W.device

    c, r = W.shape
    dtype = W.dtype
    W_fp = W.float().clone()
    X_fp = X.float()
    target = (X_fp @ W_fp).detach()                     # (n, r) reference output

    n_groups = (c + group_size - 1) // group_size
    pad = n_groups * group_size - c
    if pad > 0:
        W_fp = torch.cat([W_fp, torch.zeros(pad, r, device=device)], dim=0)

    W_g = W_fp.view(n_groups, group_size, r)            # (G, g, r)

    # Per-group rotation parameters: thetas and alpha (channel-wise scale)
    # pairs[g_idx]: list of (n_rotations x n_pairs_per_rot) tuples
    n_pairs_per_rot = group_size // 2
    all_pairs = [
        select_independent_pairs(group_size, n_rotations, n_pairs_per_rot)
        for _ in range(n_groups)
    ]

    # Learnable parameters
    thetas_all = [
        nn.ParameterList([
            nn.Parameter(torch.zeros(len(all_pairs[g][k]), device=device))
            for k in range(n_rotations)
        ])
        for g in range(n_groups)
    ]
    alpha_all = nn.ParameterList([
        nn.Parameter(torch.ones(group_size, device=device))
        for _ in range(n_groups)
    ])

    params = []
    for g in range(n_groups):
        for k in range(n_rotations):
            params.append({'params': thetas_all[g][k], 'lr': lr})
        params.append({'params': alpha_all[g], 'lr': lr})

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr / 20)

    for step in range(n_steps):
        optimizer.zero_grad()
        W_transformed = []
        for g in range(n_groups):
            Wg = W_g[g].detach().clone()                 # (g, r)
            # Apply channel-wise scaling
            Wg = alpha_all[g].unsqueeze(1) * Wg
            # Apply K independent rotations
            for k in range(n_rotations):
                Wg = apply_givens_to_weight(Wg, all_pairs[g][k], thetas_all[g][k])
            W_transformed.append(Wg)

        W_t = torch.cat(W_transformed, dim=0)[:c]        # (c, r)

        # RTN quantisation (straight-through estimator for gradients through scale)
        with torch.no_grad():
            s, _ = absmax_scale(W_t.detach(), bits, group_size)
        W_q = quantize_dequantize(W_t, s, bits)

        # Inverse transform is applied at inference; here we minimise output error
        # Output with quantised transformed weights
        out_q = X_fp @ W_q                               # (n, r)
        loss = (out_q - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Final forward pass with no grad
    W_final = []
    with torch.no_grad():
        for g in range(n_groups):
            Wg = W_g[g].clone()
            Wg = alpha_all[g].unsqueeze(1) * Wg
            for k in range(n_rotations):
                Wg = apply_givens_to_weight(Wg, all_pairs[g][k], thetas_all[g][k])
            W_final.append(Wg)

    W_t = torch.cat(W_final, dim=0)[:c]
    s, _ = absmax_scale(W_t, bits, group_size)
    Z = round_to_grid(W_t / s, bits, clip=True).to(torch.int8)
    Q = (Z.float() * s).to(dtype)

    mse = ((X_fp @ (W_fp[:c].to(dtype).float() - Q.float())).pow(2)).mean().item()
    return Q, Z, mse, float('nan')


# ---------------------------------------------------------------------------
# Model hooks: collect calibration activations per linear layer
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    name: str
    method: str
    mse: float
    bound: float
    time_s: float
    n_params: int
    # Path to the on-disk shard for this (layer, method) pair.
    # Never held in RAM beyond the single layer being processed.
    weight_path: Optional[Path] = field(default=None, repr=False)


def _save_weight(Q: Tensor, path: Path) -> None:
    """Save a dequantized weight tensor to disk as fp16."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(Q.to(torch.float16).cpu(), path)


def _load_weight(path: Path, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Load a weight tensor previously saved by _save_weight."""
    return torch.load(path, map_location='cpu', weights_only=True).to(dtype=dtype, device=device)


class ActivationCollector:
    """Forward hook to collect input activations for a single nn.Linear."""

    def __init__(self, n_calib: int, seq_len: int):
        self.n_calib = n_calib
        self.seq_len = seq_len
        self.activations: List[Tensor] = []
        self.handle = None

    def hook(self, module: nn.Module, inp, out):
        x = inp[0].detach()
        # x: (batch, seq, c) or (batch*seq, c)
        if x.dim() == 3:
            b, s, c = x.shape
            x = x.reshape(b * s, c)
        self.activations.append(x.cpu())

    def register(self, module: nn.Module):
        self.handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

    def get_X(self, device) -> Tensor:
        X = torch.cat(self.activations, dim=0)
        # Subsample to at most n_calib * seq_len rows
        max_rows = self.n_calib * self.seq_len
        if X.shape[0] > max_rows:
            idx = torch.randperm(X.shape[0])[:max_rows]
            X = X[idx]
        return X.to(device)


# ---------------------------------------------------------------------------
# Quantization order: act-order (descending Hessian diagonal)
# ---------------------------------------------------------------------------

def act_order_perm(H: Tensor) -> Tensor:
    """Return permutation that sorts Hessian diagonal in descending order."""
    return torch.argsort(H.diagonal(), descending=True)


# ---------------------------------------------------------------------------
# Per-layer benchmark
# ---------------------------------------------------------------------------

def benchmark_layer(
    name: str,
    W: Tensor,
    X: Tensor,
    bits: int,
    group_size: int,
    methods: List[str],
    device: torch.device,
    weight_dir: Path,
    damp_factor: float = 0.01,
    paroquant_steps: int = 100,
) -> List[LayerResult]:
    # Quantized weights are written to weight_dir immediately and not kept in RAM.
    results = []
    c, r = W.shape
    W = W.to(device).float()
    X = X.to(device).float()

    # Shared scale (MSE, computed once for this layer)
    scale, _ = mse_scale(W, bits, group_size)
    scale = scale.to(device)

    # Act-order permutation (shared across GPTQ variants)
    H_diag = (X.T @ X).diagonal()
    perm = torch.argsort(H_diag, descending=True)

    for method in methods:
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()

        if method == 'rtn':
            Q = quantize_dequantize(W, scale, bits)
            Z = round_to_grid(W / scale, bits).to(torch.int8)
            mse = (X @ (W - Q)).pow(2).mean().item()
            bound = float('nan')

        elif method == 'gptq':
            Q, Z, mse, bound = gptq_quantize_layer(
                W, X, scale, bits, perm=perm, damp_factor=damp_factor
            )

        elif method == 'qr_gptq':
            Q, Z, mse, bound = qr_gptq_quantize_layer(
                W, X, scale, bits, perm=perm, damp_factor=damp_factor
            )

        elif method == 'paroquant':
            Q, Z, mse, bound = paroquant_quantize_layer(
                W, X, scale, bits,
                group_size=group_size,
                n_steps=paroquant_steps,
                device=device,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.perf_counter()

        # Persist to disk; drop from RAM immediately.
        safe_name = name.replace('/', '__').replace('.', '_')
        w_path = weight_dir / f"{safe_name}__{method}.pt"
        _save_weight(Q, w_path)
        del Q, Z
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        results.append(LayerResult(
            name=name,
            method=method,
            mse=mse,
            bound=bound,
            time_s=t1 - t0,
            n_params=c * r,
            weight_path=w_path,
        ))

    # Release activation matrix for this layer now that all methods are done.
    del W, X, scale, perm
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    tokenizer,
    calib_device: torch.device,
    dataset_name: str = "salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    n_tokens: int = 2048 * 64,
    seq_len: int = 2048,
) -> float:
    """
    Compute WikiText-2 test-split perplexity.

    calib_device is the device holding the model's first parameter (i.e. the
    device that expects the input token tensor).  The function does NOT call
    model.to() because the model may already be sharded across GPU+CPU via
    device_map='auto'; calling .to() on a sharded model either raises or
    silently no-ops.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; skipping perplexity evaluation.")
        return float('nan')

    data = load_dataset(dataset_name, dataset_config, split='test')
    text = '\n\n'.join(data['text'])
    tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    tokens = tokens[:n_tokens]

    model.eval()
    total_nll = 0.0
    n_seqs = len(tokens) // seq_len

    for i in range(n_seqs):
        # Send token ids only to the embedding layer's device; accelerate
        # dispatches subsequent tensors between layers automatically.
        chunk = tokens[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(calib_device)
        with torch.amp.autocast('cuda', enabled=(calib_device.type == 'cuda')):
            out = model(chunk, labels=chunk)
        total_nll += out.loss.item()

    ppl = math.exp(total_nll / n_seqs)
    return ppl


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    model_name: str,
    bits: int = 4,
    group_size: int = 128,
    n_calib: int = 128,
    seq_len: int = 2048,
    methods: List[str] = None,
    device_str: str = 'cuda',
    damp_factor: float = 0.01,
    paroquant_steps: int = 100,
    eval_ppl: bool = True,
    max_layers: Optional[int] = None,
    weight_dir: Optional[Path] = None,
):
    import transformers

    if methods is None:
        methods = ['rtn', 'gptq', 'qr_gptq', 'paroquant']

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Model:   {model_name}")
    print(f"Bits:    {bits},  Group size: {group_size}")
    print(f"Methods: {methods}")
    print(f"Device:  {device}")
    print(f"{'='*70}\n")

    # --- load model ---
    print("Loading model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Use device_map='auto' so the model is sharded across available GPUs/CPU
    # as needed.  On a single GPU with enough VRAM (≥24 GB for 8B fp16) the
    # entire model lands on cuda:0.  On smaller cards the accelerate dispatcher
    # handles CPU offload automatically via max_memory.
    load_kwargs = dict(torch_dtype=torch.float16)
    if device.type == 'cuda':
        # Reserve a small host-memory headroom for activations; let accelerate
        # fill the rest of GPU VRAM with model weights.
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        # Leave 2 GiB on GPU for activations and workspace.
        gpu_budget_gib = max(1, (gpu_mem - 2 * 1024 ** 3) // (1024 ** 3))
        load_kwargs['device_map'] = 'auto'
        load_kwargs['max_memory'] = {
            int(device.index or 0): f"{gpu_budget_gib}GiB",
            'cpu': '48GiB',
        }
    else:
        load_kwargs['device_map'] = 'cpu'

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Identify which physical device hosts the model's embedding layer so we
    # can send calibration tokens to the same device.  With device_map='auto'
    # this is always a GPU if one is available; fall back to the benchmark device.
    try:
        calib_device = next(model.parameters()).device
    except StopIteration:
        calib_device = device

    print(f"Model loaded. Embedding device: {calib_device}")

    # --- calibration data ---
    print("Preparing calibration data...")
    try:
        from datasets import load_dataset
        calib_data = load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
        calib_text = '\n\n'.join(calib_data['text'])
        calib_tokens = tokenizer(calib_text, return_tensors='pt').input_ids[0]
        # Build n_calib sequences of length seq_len
        n_avail = len(calib_tokens) // seq_len
        n_use = min(n_calib, n_avail)
        calib_inputs = torch.stack([
            calib_tokens[i * seq_len:(i + 1) * seq_len]
            for i in range(n_use)
        ])                                               # (n_use, seq_len)
    except Exception as e:
        print(f"Warning: could not load calibration data ({e}); using random tokens.")
        calib_inputs = torch.randint(0, 32000, (n_calib, seq_len))
        n_use = n_calib

    # --- register hooks and collect activations ---
    print("Collecting calibration activations...")
    collectors: Dict[str, ActivationCollector] = {}
    linear_layers: Dict[str, nn.Linear] = {}

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            col = ActivationCollector(n_calib, seq_len)
            col.register(mod)
            collectors[name] = col
            linear_layers[name] = mod

    with torch.no_grad():
        for i in range(min(n_use, n_calib)):
            # Send tokens to the device that holds the input embeddings.
            # The hook captures activations on whatever device each layer runs
            # on, then immediately offloads them to CPU to avoid GPU OOM.
            inp = calib_inputs[i:i+1].to(calib_device)
            model(inp)

    for col in collectors.values():
        col.remove()

    # --- benchmark each layer ---
    all_results: List[LayerResult] = []
    layer_names = list(linear_layers.keys())
    if max_layers is not None:
        layer_names = layer_names[:max_layers]

    # Temporary directory for per-layer quantized weight shards.
    _own_weight_dir = weight_dir is None
    if _own_weight_dir:
        weight_dir = Path(tempfile.mkdtemp(prefix='qr_gptq_weights_'))
    else:
        weight_dir = Path(weight_dir)
        weight_dir.mkdir(parents=True, exist_ok=True)
    print(f"Weight shards: {weight_dir}")

    print(f"\nQuantizing {len(layer_names)} layers with {len(methods)} methods...\n")
    print(f"{'Layer':<50} {'Method':<12} {'MSE':>12} {'Bound':>14} {'Time(s)':>10}")
    print('-' * 100)

    for layer_name in layer_names:
        mod = linear_layers[layer_name]
        W = mod.weight.data.T.contiguous()               # (in, out) -> (c, r)
        X = collectors[layer_name].get_X(device)

        if X.shape[0] < 16:
            # Too few activations collected; skip
            continue

        layer_results = benchmark_layer(
            name=layer_name,
            W=W,
            X=X,
            bits=bits,
            group_size=group_size,
            methods=methods,
            device=device,
            weight_dir=weight_dir,
            damp_factor=damp_factor,
            paroquant_steps=paroquant_steps,
        )

        # Free collector activations once all methods have consumed them.
        collectors[layer_name].activations.clear()

        for res in layer_results:
            all_results.append(res)
            bound_str = f"{res.bound:.4e}" if not math.isnan(res.bound) else "      n/a"
            print(
                f"{res.name:<50} {res.method:<12} "
                f"{res.mse:>12.4e} {bound_str:>14} {res.time_s:>10.2f}"
            )

    # --- summary statistics ---
    print(f"\n{'='*70}")
    print("SUMMARY (mean across layers)")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'Mean MSE':>14} {'Mean Bound':>14} {'Mean Time(s)':>14} {'Total params':>14}")
    print('-' * 70)

    method_results: Dict[str, List[LayerResult]] = {}
    for res in all_results:
        method_results.setdefault(res.method, []).append(res)

    for method, rlist in method_results.items():
        mean_mse = sum(r.mse for r in rlist) / len(rlist)
        valid_bounds = [r.bound for r in rlist if not math.isnan(r.bound)]
        mean_bound = sum(valid_bounds) / len(valid_bounds) if valid_bounds else float('nan')
        mean_time = sum(r.time_s for r in rlist) / len(rlist)
        total_params = sum(r.n_params for r in rlist)
        bound_str = f"{mean_bound:.4e}" if not math.isnan(mean_bound) else "         n/a"
        print(
            f"{method:<12} {mean_mse:>14.4e} {bound_str:>14} "
            f"{mean_time:>14.2f} {total_params:>14,}"
        )

    # --- MSE reduction relative to RTN ---
    if 'rtn' in method_results:
        rtn_mse_map = {r.name: r.mse for r in method_results['rtn']}
        print(f"\n{'='*70}")
        print("MSE reduction vs RTN baseline")
        print(f"{'='*70}")
        print(f"{'Method':<12} {'Reduction (x)':>14} {'Reduction (dB)':>16}")
        print('-' * 44)
        for method, rlist in method_results.items():
            if method == 'rtn':
                continue
            ratios = [
                rtn_mse_map[r.name] / r.mse
                for r in rlist
                if r.name in rtn_mse_map and r.mse > 0
            ]
            if ratios:
                mean_ratio = sum(ratios) / len(ratios)
                mean_db = 10 * math.log10(mean_ratio)
                print(f"{method:<12} {mean_ratio:>14.2f}x {mean_db:>15.2f} dB")

    # --- Condition number diagnostic ---
    print(f"\n{'='*70}")
    print("Numerical stability: condition number of H vs X")
    print(f"{'='*70}")
    diag_layers = layer_names[:5]
    for lname in diag_layers:
        if lname not in collectors:
            continue
        X_d = collectors[lname].get_X(device).float()
        if X_d.shape[0] < X_d.shape[1]:
            continue
        H = X_d.T @ X_d
        lam = damp_factor / X_d.shape[1] * H.trace() / X_d.shape[1]
        H.diagonal().add_(lam)
        sv_X = torch.linalg.svdvals(X_d)
        sv_H = torch.linalg.svdvals(H)
        kappa_X = (sv_X.max() / sv_X[sv_X > 1e-10].min()).item()
        kappa_H = (sv_H.max() / sv_H[sv_H > 1e-10].min()).item()
        print(f"  {lname:<50}  kappa(X)={kappa_X:.2e}  kappa(H)={kappa_H:.2e}  ratio={kappa_H/kappa_X:.2f}x")

    # --- optional perplexity ---
    # For each method, load quantized weights from disk one layer at a time,
    # run perplexity, then restore originals (also read from disk one layer
    # at a time).  No bulk in-RAM dicts are created.
    if eval_ppl:
        print(f"\n{'='*70}")
        print("Perplexity on WikiText-2 test split")
        print(f"{'='*70}")
        print(f"{'Method':<12} {'Perplexity':>12}")
        print('-' * 26)

        # O(1) lookup: (layer_name, method) -> Path
        path_index: Dict[Tuple[str, str], Path] = {
            (r.name, r.method): r.weight_path
            for r in all_results
            if r.weight_path is not None
        }

        for method in methods:
            # Write quantized weights, one layer at a time.
            for name, mod in linear_layers.items():
                key = (name, method)
                if key not in path_index:
                    continue
                Q = _load_weight(
                    path_index[key],
                    dtype=mod.weight.dtype,
                    device=mod.weight.data.device,
                )  # (c, r)
                mod.weight.data.copy_(Q.T)  # (r, c)
                del Q

            ppl = compute_perplexity(
                model, tokenizer,
                calib_device=calib_device,
                n_tokens=2048 * 64,
                seq_len=seq_len,
            )
            print(f"{method:<12} {ppl:>12.3f}")

            # Restore originals one layer at a time from disk.
            # Original shards are written on the first restore pass.
            for name, mod in linear_layers.items():
                orig_key = (name, '__orig__')
                if orig_key not in path_index:
                    orig_safe = name.replace('/', '__').replace('.', '_')
                    orig_path = weight_dir / f"{orig_safe}____orig__.pt"
                    if not orig_path.exists():
                        _save_weight(mod.weight.data.T.contiguous(), orig_path)
                    path_index[orig_key] = orig_path
                orig = _load_weight(
                    path_index[orig_key],
                    dtype=mod.weight.dtype,
                    device=mod.weight.data.device,
                )  # (c, r)
                mod.weight.data.copy_(orig.T)
                del orig

    # --- clean up temporary shard directory if we created it ---
    if _own_weight_dir:
        shutil.rmtree(weight_dir, ignore_errors=True)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark QR-GPTQ vs GPTQ vs ParoQuant on LLaMA-3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='HuggingFace model ID')
    parser.add_argument('--bits', type=int, default=4,
                        help='Quantization bit width')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Per-group quantization block size')
    parser.add_argument('--n-calib', type=int, default=128,
                        help='Number of calibration sequences')
    parser.add_argument('--seq-len', type=int, default=2048,
                        help='Calibration sequence length')
    parser.add_argument('--methods', nargs='+',
                        default=['rtn', 'gptq', 'qr_gptq', 'paroquant'],
                        choices=['rtn', 'gptq', 'qr_gptq', 'paroquant'],
                        help='Methods to benchmark')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--damp-factor', type=float, default=0.01,
                        help='Hessian damping ratio lambda')
    parser.add_argument('--paroquant-steps', type=int, default=100,
                        help='Gradient steps for ParoQuant Stage 1')
    parser.add_argument('--no-ppl', action='store_true',
                        help='Skip perplexity evaluation')
    parser.add_argument('--max-layers', type=int, default=None,
                        help='Limit to first N layers (for fast testing)')
    parser.add_argument('--weight-dir', type=str, default=None,
                        help='Directory for quantized weight shards '
                             '(default: auto temp dir, deleted on exit)')
    return parser.parse_args()


def main():
    args = parse_args()

    models_to_run = [args.model]

    # If the user passed the generic name, expand to 1B and 8B
    if args.model in ('llama3', 'llama-3'):
        models_to_run = [
            'meta-llama/Llama-3.2-1B',
            'meta-llama/Meta-Llama-3-8B',
        ]

    for model_name in models_to_run:
        run_benchmark(
            model_name=model_name,
            bits=args.bits,
            group_size=args.group_size,
            n_calib=args.n_calib,
            seq_len=args.seq_len,
            methods=args.methods,
            device_str=args.device,
            damp_factor=args.damp_factor,
            paroquant_steps=args.paroquant_steps,
            eval_ppl=not args.no_ppl,
            max_layers=args.max_layers,
            weight_dir=Path(args.weight_dir) if args.weight_dir else None,
        )


if __name__ == '__main__':
    main()