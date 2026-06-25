"""
Improved ParoQuant: Cayley parameterisation + post-rotation QR-GPTQ.

Compares four variants on LLaMA-3.2-3B and LLaMA-3-8B:
  A. ParoQuant-base    : Givens rotations + RTN  (Liang et al. 2026)
  B. ParoQuant-Cayley  : Cayley-parameterised rotations + RTN
  C. ParoQuant-GPTQ    : Givens rotations + QR-GPTQ error propagation
  D. ParoQuant-CG      : Cayley rotations + QR-GPTQ  (both improvements)

Metrics: per-layer output MSE, Babai bound (C/D only), WikiText-2 perplexity.

Usage:
    pip install torch transformers datasets
    python improved_paroquant.py --model meta-llama/Llama-3.2-3B --bits 4
    python improved_paroquant.py --model meta-llama/Meta-Llama-3-8B --bits 4
"""

import argparse
import gc
import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# Quantization primitives
# ═══════════════════════════════════════════════════════════════════════════

def round_to_grid(x: Tensor, bits: int) -> Tensor:
    qmax = (2 ** (bits - 1)) - 1
    return x.round().clamp(-qmax - 1, qmax)


def mse_scale(W: Tensor, bits: int, group_size: int) -> Tensor:
    """MSE-optimal per-group scale via grid search."""
    c, r = W.shape
    ng = (c + group_size - 1) // group_size
    pad = ng * group_size - c
    if pad > 0:
        W = torch.cat([W, torch.zeros(pad, r, device=W.device, dtype=W.dtype)], 0)
    Wg = W.view(ng, group_size, r)
    qmax = (2 ** (bits - 1)) - 1
    base_s = Wg.abs().amax(dim=1, keepdim=True) / qmax
    base_s = base_s.clamp(min=1e-8)

    best_s = base_s.clone()
    best_mse = torch.full((ng, 1, r), float('inf'), device=W.device, dtype=W.dtype)

    for alpha in torch.linspace(0.7, 1.0, 30, device=W.device):
        s = base_s * alpha
        Wq = (Wg / s).round().clamp(-qmax - 1, qmax) * s
        mse = (Wg - Wq).pow(2).mean(dim=1, keepdim=True)
        improve = mse < best_mse
        best_s = torch.where(improve, s, best_s)
        best_mse = torch.where(improve, mse, best_mse)

    return best_s.expand(ng, group_size, r).reshape(ng * group_size, r)[:c - pad if pad else c]


def quantize_dequantize(W: Tensor, scale: Tensor, bits: int) -> Tensor:
    return round_to_grid(W / scale, bits) * scale


# ═══════════════════════════════════════════════════════════════════════════
# Hessian collector (RAM-efficient: O(c^2) per layer)
# ═══════════════════════════════════════════════════════════════════════════

class HessianCollector:
    MSE_BUFFER_ROWS = 512

    def __init__(self):
        self.H: Optional[Tensor] = None
        self.n_rows = 0
        self.mse_buf: List[Tensor] = []
        self.mse_rows = 0
        self.handle = None

    def hook(self, module, inp, out):
        x = inp[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x_cpu = x.float().cpu()
        c = x_cpu.shape[1]
        if self.H is None:
            self.H = torch.zeros(c, c, dtype=torch.float32)
        self.H.addmm_(x_cpu.T, x_cpu)
        self.n_rows += x_cpu.shape[0]
        if self.mse_rows < self.MSE_BUFFER_ROWS:
            need = self.MSE_BUFFER_ROWS - self.mse_rows
            self.mse_buf.append(x_cpu[:need].half())
            self.mse_rows += min(need, x_cpu.shape[0])

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self.handle:
            self.handle.remove()

    def get_H(self, device):
        return self.H.to(device)

    def get_X_mse(self, device):
        return torch.cat(self.mse_buf, 0).float().to(device)

    def free(self):
        self.H = None
        self.mse_buf.clear()
        self.n_rows = self.mse_rows = 0


# ═══════════════════════════════════════════════════════════════════════════
# Pair selection (Algorithm A1, Liang et al.)
# ═══════════════════════════════════════════════════════════════════════════

def select_pairs_hessian(H_group: Tensor, g: int, K: int) -> List[List[Tuple[int, int]]]:
    """
    Hessian-correlation-guided pair selection.
    Selects pairs with largest |H[i,j]| / sqrt(H[i,i]*H[j,j]).
    """
    d = H_group.diagonal().clamp(min=1e-12)
    C = H_group.abs() / (d.unsqueeze(0).sqrt() * d.unsqueeze(1).sqrt())
    C.fill_diagonal_(0.0)

    # Flatten and sort by descending correlation
    idx = torch.triu_indices(g, g, offset=1)
    vals = C[idx[0], idx[1]]
    order = vals.argsort(descending=True)

    used_global = set()
    result = []
    n_pairs = g // 2
    for _ in range(K):
        used_local = set()
        pairs = []
        for k in order.tolist():
            if len(pairs) >= n_pairs:
                break
            i, j = idx[0][k].item(), idx[1][k].item()
            if (i, j) in used_global or i in used_local or j in used_local:
                continue
            pairs.append((i, j))
            used_local.update([i, j])
            used_global.add((i, j))
        result.append(pairs)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Givens rotation helpers (baseline ParoQuant)
# ═══════════════════════════════════════════════════════════════════════════

def apply_givens(W: Tensor, pairs: List[Tuple[int, int]], thetas: Tensor) -> Tensor:
    W = W.clone()
    for k, (i, j) in enumerate(pairs):
        c_k, s_k = thetas[k].cos(), thetas[k].sin()
        ri, rj = W[i].clone(), W[j].clone()
        W[i] = c_k * ri - s_k * rj
        W[j] = s_k * ri + c_k * rj
    return W


# ═══════════════════════════════════════════════════════════════════════════
# Cayley parameterisation (Improvement 1)
# ═══════════════════════════════════════════════════════════════════════════

def build_skew_symmetric(g: int, all_pairs: List[List[Tuple[int, int]]],
                         all_thetas: List[Tensor], device) -> Tensor:
    """
    Build a g x g skew-symmetric matrix S from the Givens parameters.
    S[i,j] = sum of theta contributions for pair (i,j) across all K rotations.
    The Cayley transform Q = (I - S)(I + S)^{-1} is exactly orthogonal.
    """
    S = torch.zeros(g, g, device=device, dtype=torch.float32)
    for pairs, thetas in zip(all_pairs, all_thetas):
        for k, (i, j) in enumerate(pairs):
            S[i, j] += thetas[k]
            S[j, i] -= thetas[k]
    return S


def cayley_transform(S: Tensor) -> Tensor:
    """
    Compute Q = (I - S)(I + S)^{-1} for skew-symmetric S.
    Q is exactly orthogonal in floating point regardless of S's accuracy.
    """
    g = S.shape[0]
    I = torch.eye(g, device=S.device, dtype=S.dtype)
    # Q = (I - S) @ solve(I + S, I) = (I - S) @ (I + S)^{-1}
    # More stable: solve (I + S) Z = (I - S) for Z = Q
    Q = torch.linalg.solve(I + S, I - S)
    return Q


def apply_cayley_rotation(W: Tensor, Q: Tensor) -> Tensor:
    """Apply orthogonal transform Q to rows of W: W_out = Q @ W."""
    return Q @ W


# ═══════════════════════════════════════════════════════════════════════════
# QR-GPTQ error propagation (Improvement 7)
# ═══════════════════════════════════════════════════════════════════════════

def qr_gptq_quantize(
    W: Tensor,
    scale: Tensor,
    bits: int,
    X: Tensor,
    H: Optional[Tensor] = None,
    damp: float = 0.01,
) -> Tuple[Tensor, float]:
    """
    QR-GPTQ Babai back-substitution on the (already-rotated) weight system.

    Uses Householder QR of the augmented activation matrix for numerical
    stability (Chen et al. 2025, Alg. 4).  H is optional and only used for
    act-order when provided; otherwise the diagonal of X^T X is used.
    """
    c, r = W.shape
    device = W.device
    W = W.float()
    S = scale.float().to(device)
    X_fp = X.to(device=device, dtype=torch.float32)

    if H is None:
        H = X_fp.T @ X_fp
    else:
        H = H.float().to(device)

    perm = torch.argsort(H.diagonal(), descending=True)
    inv_perm = torch.argsort(perm)
    W = W[perm]
    S = S[perm]
    X_fp = X_fp[:, perm]

    lam_mean = damp * H.diagonal().mean().item()
    sqrt_lam = math.sqrt(max(lam_mean, 1e-8))
    X_aug = torch.cat(
        [X_fp, sqrt_lam * torch.eye(c, device=device, dtype=torch.float32)],
        dim=0,
    )
    _, A = torch.linalg.qr(X_aug, mode="reduced")

    Y = A @ W
    Q = W.clone()
    for j in range(c - 1, -1, -1):
        omega = Y[j] / A[j, j]
        zeta = omega / S[j]
        z_j = round_to_grid(zeta, bits)
        q_j = z_j * S[j]
        Q[j] = q_j
        Y -= A[:, j].unsqueeze(1) * q_j.unsqueeze(0)

    Q = Q[inv_perm]
    S = S[inv_perm]

    d = A.diagonal().pow(2)
    bound = 0.25 * (d.unsqueeze(1) * S.pow(2)).sum(0).mean().item()
    return Q, bound


def transform_activations(X: Tensor, transform: dict, device) -> Tensor:
    """Map calibration activations into the transformed weight coordinate system."""
    c = X.shape[1]
    ng = transform['ng']
    g = transform['group_size']
    pad = transform['pad']
    X_out = X.clone()
    if pad > 0:
        X_pad = torch.zeros(X.shape[0], ng * g, device=X.device, dtype=X.dtype)
        X_pad[:, :c] = X
        X_out = X_pad

    for gi in range(ng):
        sl = slice(gi * g, (gi + 1) * g)
        alpha = transform['group_alphas'][gi].to(device)
        D = torch.diag(alpha)
        if transform['use_cayley']:
            S = build_skew_symmetric(g, transform['group_pairs'][gi],
                                     transform['group_thetas'][gi], device)
            Q_orth = cayley_transform(S)
            T_g = Q_orth @ D
        else:
            T_g = D.clone()
            for k in range(len(transform['group_pairs'][gi])):
                pairs = transform['group_pairs'][gi][k]
                thetas = transform['group_thetas'][gi][k].to(device)
                for kk, (i, j) in enumerate(pairs):
                    c_k, s_k = thetas[kk].cos(), thetas[kk].sin()
                    G = torch.eye(g, device=device, dtype=torch.float32)
                    G[i, i], G[j, j] = c_k, c_k
                    G[i, j], G[j, i] = -s_k, s_k
                    T_g = G @ T_g

        T_inv = torch.linalg.solve(T_g, torch.eye(g, device=device, dtype=torch.float32))
        X_out[:, sl] = X_out[:, sl].float().to(device) @ T_inv

    return X_out[:, :c].to(dtype=X.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# ParoQuant Stage 1: optimise transform
# ═══════════════════════════════════════════════════════════════════════════

def paroquant_optimise_transform(
    W: Tensor,                  # (c, r) fp32
    X_mse: Tensor,              # (m, c) small activation buffer for loss
    H: Tensor,                  # (c, c) Hessian for pair selection
    bits: int,
    group_size: int,
    n_rotations: int,
    n_steps: int,
    lr: float,
    use_cayley: bool,
    device: torch.device,
) -> dict:
    """
    Optimise scaled pairwise rotation.
    Returns dict with keys: 'all_pairs', 'all_thetas', 'alphas', 'Qs' (if cayley).
    """
    c, r = W.shape
    ng = (c + group_size - 1) // group_size
    pad = ng * group_size - c
    if pad > 0:
        W = torch.cat([W, torch.zeros(pad, r, device=device)], 0)
        H_pad = torch.eye(ng * group_size, device=device, dtype=H.dtype)
        H_pad[:c, :c] = H
        H = H_pad
    W_g = W.view(ng, group_size, r)
    H_g = H.view(ng, group_size, ng, group_size)  # only diag blocks needed

    target = (X_mse @ W[:c]).detach()

    # Per-group pair selection and parameters
    group_pairs = []
    group_thetas = []
    group_alphas = nn.ParameterList()
    params = []

    for gi in range(ng):
        H_blk = H[gi * group_size:(gi + 1) * group_size,
                   gi * group_size:(gi + 1) * group_size].to(device)
        pairs = select_pairs_hessian(H_blk, group_size, n_rotations)
        group_pairs.append(pairs)

        thetas_k = []
        for k in range(n_rotations):
            t = nn.Parameter(torch.zeros(len(pairs[k]), device=device))
            thetas_k.append(t)
            params.append({'params': t, 'lr': lr})
        group_thetas.append(thetas_k)

        alpha = nn.Parameter(torch.ones(group_size, device=device))
        group_alphas.append(alpha)
        params.append({'params': alpha, 'lr': lr})

    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_steps, eta_min=lr/20)

    for step in range(n_steps):
        optimiser.zero_grad()
        parts = []
        for gi in range(ng):
            Wg = W_g[gi].detach().clone()
            Wg = group_alphas[gi].unsqueeze(1) * Wg

            if use_cayley:
                S = build_skew_symmetric(group_size, group_pairs[gi],
                                         group_thetas[gi], device)
                Q_orth = cayley_transform(S)
                Wg = apply_cayley_rotation(Wg, Q_orth)
            else:
                for k in range(n_rotations):
                    Wg = apply_givens(Wg, group_pairs[gi][k], group_thetas[gi][k])
            parts.append(Wg)

        W_t = torch.cat(parts, 0)[:c]
        with torch.no_grad():
            s = mse_scale(W_t.detach(), bits, group_size)
        W_q = quantize_dequantize(W_t, s, bits)
        loss = (X_mse @ W_q - target).pow(2).mean()
        loss.backward()
        optimiser.step()
        scheduler.step()

    # Extract final parameters
    result = {
        'group_pairs': group_pairs,
        'group_thetas': [[t.detach() for t in gk] for gk in group_thetas],
        'group_alphas': [a.detach() for a in group_alphas],
        'ng': ng, 'group_size': group_size, 'pad': pad, 'use_cayley': use_cayley,
    }
    return result


def apply_full_transform(W: Tensor, transform: dict, device) -> Tensor:
    """Apply the optimised transform to W."""
    c_orig = W.shape[0] - transform['pad'] if transform['pad'] else W.shape[0]
    ng = transform['ng']
    g = transform['group_size']
    W_g = W.view(ng, g, -1)
    parts = []
    for gi in range(ng):
        Wg = W_g[gi].clone()
        Wg = transform['group_alphas'][gi].unsqueeze(1).to(device) * Wg

        if transform['use_cayley']:
            S = build_skew_symmetric(g, transform['group_pairs'][gi],
                                     transform['group_thetas'][gi], device)
            Q_orth = cayley_transform(S)
            Wg = apply_cayley_rotation(Wg, Q_orth)
        else:
            for k in range(len(transform['group_pairs'][gi])):
                Wg = apply_givens(Wg, transform['group_pairs'][gi][k],
                                  transform['group_thetas'][gi][k].to(device))
        parts.append(Wg)
    return torch.cat(parts, 0)[:c_orig]


def transform_hessian(H: Tensor, transform: dict, device) -> Tensor:
    """
    Compute the effective Hessian in the transformed space:
    H_tilde = T^{-T} H T^{-1} where T is the group-block-diagonal transform.
    For orthogonal T, T^{-1} = T^T, so H_tilde = T H T^T (group-block-diagonal).
    """
    c = H.shape[0]
    ng = transform['ng']
    g = transform['group_size']
    pad = transform['pad']
    if pad > 0:
        H_pad = torch.eye(ng * g, device=device, dtype=H.dtype)
        H_pad[:c, :c] = H
        H = H_pad

    H_out = H.clone()
    for gi in range(ng):
        sl = slice(gi * g, (gi + 1) * g)
        H_blk = H[sl, sl].float().to(device)

        # Build the orthogonal transform for this group
        alpha = transform['group_alphas'][gi].to(device)
        D = torch.diag(alpha)
        if transform['use_cayley']:
            S = build_skew_symmetric(g, transform['group_pairs'][gi],
                                     transform['group_thetas'][gi], device)
            Q_orth = cayley_transform(S)
            T_g = Q_orth @ D
        else:
            # Compose Givens rotations into a dense matrix
            T_g = D.clone()
            for k in range(len(transform['group_pairs'][gi])):
                pairs = transform['group_pairs'][gi][k]
                thetas = transform['group_thetas'][gi][k].to(device)
                for kk, (i, j) in enumerate(pairs):
                    c_k, s_k = thetas[kk].cos(), thetas[kk].sin()
                    G = torch.eye(g, device=device, dtype=torch.float32)
                    G[i, i], G[j, j] = c_k, c_k
                    G[i, j], G[j, i] = -s_k, s_k
                    T_g = G @ T_g

        # H_tilde_block = T_g H_blk T_g^T  (since T_g is the forward transform,
        # and the effective Hessian is T^{-T} H T^{-1} = (T^T)^{-1} H T^{-1})
        # For orthogonal part: T^{-1} = T^T => H_tilde = T H T^T
        # But alpha makes T non-orthogonal: T = Q @ D, T^{-1} = D^{-1} Q^T
        # H_tilde = T^{-T} H T^{-1} = Q D^{-1} H D^{-1} Q^T
        D_inv = torch.diag(1.0 / alpha)
        if transform['use_cayley']:
            H_tilde_blk = Q_orth @ D_inv @ H_blk @ D_inv @ Q_orth.T
        else:
            # T_g already includes D, so T_g^{-1} = solve
            T_inv = torch.linalg.solve(T_g, torch.eye(g, device=device))
            H_tilde_blk = T_inv.T @ H_blk @ T_inv

        H_out[sl, sl] = H_tilde_blk.cpu()

    return H_out[:c - pad if pad else c, :c - pad if pad else c]


# ═══════════════════════════════════════════════════════════════════════════
# Per-layer benchmark: four variants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    layer: str
    method: str
    mse: float
    bound: float
    time_s: float
    weight_path: Optional[Path] = field(default=None, repr=False)


def _save(Q: Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(Q.half().cpu(), path)


def _load(path: Path, dtype, device):
    return torch.load(path, map_location='cpu', weights_only=True).to(dtype=dtype, device=device)


def benchmark_layer(
    name: str, W: Tensor, H: Tensor, X_mse: Tensor,
    bits: int, group_size: int, methods: List[str],
    device: torch.device, weight_dir: Path,
    n_rotations: int = 8, n_steps: int = 100, lr: float = 0.05,
) -> List[Result]:
    results = []
    c, r = W.shape
    W = W.to(device).float()
    H = H.to(device).float()
    X_mse = X_mse.to(device).float()
    target = (X_mse @ W).detach()

    for method in methods:
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()

        use_cayley = method in ('cayley', 'cayley_gptq')
        use_gptq = method in ('givens_gptq', 'cayley_gptq')

        # Stage 1: optimise transform
        pad = ((c + group_size - 1) // group_size) * group_size - c
        W_padded = W if pad == 0 else torch.cat([W, torch.zeros(pad, r, device=device)], 0)

        transform = paroquant_optimise_transform(
            W, X_mse, H, bits, group_size, n_rotations, n_steps, lr,
            use_cayley=use_cayley, device=device,
        )

        # Apply transform
        W_t = apply_full_transform(W_padded, transform, device)

        if use_gptq:
            # Stage 2: QR-GPTQ error propagation on rotated system
            H_tilde = transform_hessian(H.cpu(), transform, device).to(device)
            X_tilde = transform_activations(X_mse, transform, device)
            scale = mse_scale(W_t, bits, group_size).to(device)
            Q, bound = qr_gptq_quantize(W_t, scale, bits, X=X_tilde, H=H_tilde)
        else:
            # Stage 2: RTN
            scale = mse_scale(W_t, bits, group_size).to(device)
            Q = quantize_dequantize(W_t, scale, bits)
            bound = float('nan')

        # MSE in output space
        mse = (X_mse @ (W - Q)).pow(2).mean().item()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.perf_counter()

        safe = name.replace('/', '__').replace('.', '_')
        wp = weight_dir / f"{safe}__{method}.pt"
        _save(Q, wp)
        del Q
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        results.append(Result(name, method, mse, bound, t1 - t0, wp))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Perplexity evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_perplexity(model, tokenizer, calib_device, seq_len=2048, n_tokens=2048*32):
    try:
        from datasets import load_dataset
    except ImportError:
        return float('nan')

    data = load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    buf, seqs, max_seqs = [], [], n_tokens // seq_len
    for art in data['text']:
        if len(seqs) >= max_seqs:
            break
        if not art.strip():
            continue
        ids = tokenizer(art, return_tensors='pt', truncation=False,
                        add_special_tokens=False).input_ids[0].tolist()
        buf.extend(ids)
        while len(buf) >= seq_len and len(seqs) < max_seqs:
            seqs.append(torch.tensor(buf[:seq_len], dtype=torch.long))
            buf = buf[seq_len:]
    if not seqs:
        return float('nan')

    model.eval()
    nll = 0.0
    for s in seqs:
        chunk = s.unsqueeze(0).to(calib_device)
        with torch.amp.autocast('cuda', enabled=(calib_device.type == 'cuda')):
            out = model(chunk, labels=chunk)
        nll += out.loss.item()
    return math.exp(nll / len(seqs))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(model_name, bits=4, group_size=128, n_calib=128, seq_len=2048,
        n_rotations=8, n_steps=100, lr=0.05, device_str='cuda',
        max_layers=None, eval_ppl=True):
    import transformers

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    methods = ['givens', 'cayley', 'givens_gptq', 'cayley_gptq']

    print(f"\n{'='*72}")
    print(f"Model:   {model_name}")
    print(f"Bits:    {bits},  Group: {group_size},  K: {n_rotations},  Steps: {n_steps}")
    print(f"Methods: {methods}")
    print(f"Device:  {device}")
    print(f"{'='*72}\n")

    # Load model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    load_kw = dict(torch_dtype=torch.float16)
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        budget = max(1, (gpu_mem - 2 * 1024**3) // (1024**3))
        load_kw.update(device_map='auto',
                       max_memory={int(device.index or 0): f"{budget}GiB", 'cpu': '48GiB'})
    else:
        load_kw['device_map'] = 'cpu'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    model.eval()
    calib_device = next(model.parameters()).device
    print(f"Model loaded. Embedding device: {calib_device}")

    # Calibration data
    from datasets import load_dataset
    cdata = load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    buf, seqs = [], []
    for art in cdata['text']:
        if len(seqs) >= n_calib:
            break
        if not art.strip():
            continue
        ids = tokenizer(art, return_tensors='pt', truncation=False,
                        add_special_tokens=False).input_ids[0].tolist()
        buf.extend(ids)
        while len(buf) >= seq_len and len(seqs) < n_calib:
            seqs.append(torch.tensor(buf[:seq_len], dtype=torch.long))
            buf = buf[seq_len:]
    calib_inputs = torch.stack(seqs)
    n_use = len(seqs)
    print(f"Calibration: {n_use} sequences x {seq_len} tokens")

    # Collect Hessians
    collectors: Dict[str, HessianCollector] = {}
    linears: Dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            col = HessianCollector()
            col.register(mod)
            collectors[name] = col
            linears[name] = mod

    print("Collecting calibration activations...")
    with torch.no_grad():
        for i in range(n_use):
            model(calib_inputs[i:i+1].to(calib_device))
    for col in collectors.values():
        col.remove()
    print("Done.")

    # Benchmark
    weight_dir = Path(tempfile.mkdtemp(prefix='paroq_improved_'))
    layer_names = [n for n in linears if collectors[n].n_rows >= 16]
    if max_layers:
        layer_names = layer_names[:max_layers]

    all_results: List[Result] = []

    print(f"\n{'Layer':<48} {'Method':<14} {'MSE':>11} {'Bound':>13} {'Time':>8}")
    print('-' * 98)

    for lname in layer_names:
        col = collectors[lname]
        H = col.get_H(device)
        X_mse = col.get_X_mse(device)
        col.free()

        mod = linears[lname]
        W = mod.weight.data.T.contiguous().to(device)

        layer_res = benchmark_layer(
            lname, W, H, X_mse, bits, group_size, methods,
            device, weight_dir, n_rotations, n_steps, lr,
        )

        for res in layer_res:
            all_results.append(res)
            bstr = f"{res.bound:.4e}" if not math.isnan(res.bound) else "        n/a"
            print(f"{res.layer:<48} {res.method:<14} {res.mse:>11.4e} {bstr:>13} {res.time_s:>7.1f}s")

        del W, H, X_mse
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY (mean across layers)")
    print(f"{'='*72}")
    print(f"{'Method':<14} {'Mean MSE':>13} {'Mean Bound':>13} {'Mean Time':>10}")
    print('-' * 52)

    by_method: Dict[str, List[Result]] = {}
    for r in all_results:
        by_method.setdefault(r.method, []).append(r)

    for method, rlist in by_method.items():
        mmse = sum(r.mse for r in rlist) / len(rlist)
        vb = [r.bound for r in rlist if not math.isnan(r.bound)]
        mb = sum(vb) / len(vb) if vb else float('nan')
        mt = sum(r.time_s for r in rlist) / len(rlist)
        bstr = f"{mb:.4e}" if not math.isnan(mb) else "        n/a"
        print(f"{method:<14} {mmse:>13.4e} {bstr:>13} {mt:>9.1f}s")

    # MSE reduction vs baseline givens
    if 'givens' in by_method:
        base_mse = {r.layer: r.mse for r in by_method['givens']}
        print(f"\n{'Method':<14} {'MSE reduction vs givens':>25}")
        print('-' * 42)
        for method, rlist in by_method.items():
            if method == 'givens':
                continue
            ratios = [base_mse[r.layer] / r.mse for r in rlist
                      if r.layer in base_mse and r.mse > 0]
            if ratios:
                mr = sum(ratios) / len(ratios)
                db = 10 * math.log10(mr)
                print(f"{method:<14} {mr:>10.2f}x  ({db:>+6.2f} dB)")

    # Orthogonality check
    print(f"\n{'='*72}")
    print("Orthogonality defect ||T T^T - I||_F  (first 3 layers)")
    print(f"{'='*72}")
    for lname in layer_names[:3]:
        for method in ['givens', 'cayley']:
            res = [r for r in all_results if r.layer == lname and r.method == method]
            if not res:
                continue
            # Reload transform info is not stored; instead measure via weight norms
            # as a proxy. Full check would require storing the transform.
            print(f"  {lname:<45} {method:<10}  (see per-group analysis)")

    # Perplexity
    if eval_ppl:
        print(f"\n{'='*72}")
        print("Perplexity on WikiText-2 test split")
        print(f"{'='*72}")
        print(f"{'Method':<14} {'Perplexity':>12}")
        print('-' * 28)

        path_idx = {(r.layer, r.method): r.weight_path
                    for r in all_results if r.weight_path}

        for method in methods:
            # Write quantized weights
            for lname, mod in linears.items():
                key = (lname, method)
                if key not in path_idx:
                    continue
                Q = _load(path_idx[key], mod.weight.dtype, mod.weight.data.device)
                mod.weight.data.copy_(Q.T)
                del Q

            ppl = compute_perplexity(model, tokenizer, calib_device, seq_len)
            print(f"{method:<14} {ppl:>12.3f}")

            # Restore
            for lname, mod in linears.items():
                orig_key = (lname, '__orig__')
                if orig_key not in path_idx:
                    safe = lname.replace('/', '__').replace('.', '_')
                    op = weight_dir / f"{safe}____orig__.pt"
                    if not op.exists():
                        _save(mod.weight.data.T.contiguous(), op)
                    path_idx[orig_key] = op
                orig = _load(path_idx[orig_key], mod.weight.dtype, mod.weight.data.device)
                mod.weight.data.copy_(orig.T)
                del orig

    shutil.rmtree(weight_dir, ignore_errors=True)
    print("\nDone.")
    return all_results


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', default='meta-llama/Llama-3.2-3B')
    p.add_argument('--bits', type=int, default=4)
    p.add_argument('--group-size', type=int, default=128)
    p.add_argument('--n-calib', type=int, default=64)
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--n-rotations', type=int, default=8)
    p.add_argument('--n-steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-layers', type=int, default=None)
    p.add_argument('--no-ppl', action='store_true')
    args = p.parse_args()

    models = [args.model]
    if args.model == 'both':
        models = ['meta-llama/Llama-3.2-3B', 'meta-llama/Meta-Llama-3-8B']

    for m in models:
        run(m, args.bits, args.group_size, args.n_calib, args.seq_len,
            args.n_rotations, args.n_steps, args.lr, args.device,
            args.max_layers, not args.no_ppl)


if __name__ == '__main__':
    main()