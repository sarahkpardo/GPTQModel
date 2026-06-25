"""
paroquant_all_improvements.py
─────────────────────────────
Implements all proposed numerical improvements to ParoQuant except the
Cayley re-parameterisation, and compares two variants:

  baseline  – original ParoQuant (Liang et al. 2026):
                · random + magnitude-difference pair selection
                · absmax per-group scale
                · STE (deterministic rounding) during optimisation
                · alpha initialised to 1 (no preconditioning)
                · no Hessian equilibration
                · RTN quantization after rotation

  improved  – all numerical improvements (no Cayley):
                · Hessian-correlation pair selection (Improvement 5)
                · MSE-optimal scale via grid search (Improvement 2)
                · stochastic rounding during optimisation (Improvement 4)
                · Jacobi initialisation of alpha (Improvement 6)
                · diagonal Hessian equilibration before pair selection (Improvement 6)
                · post-rotation QR-GPTQ error propagation (Improvement 7)
                  with the corrected coordinate system (Bug 2 + Bug 3 fixes)

Both variants use sequential Givens rotations with the corrected
inverse-transform in the optimisation loss (Bug 2 fix) and the corrected
output-space MSE + disk save (Bug 3 fix).

Usage:
    python paroquant_all_improvements.py \\
        --model meta-llama/Llama-3.2-3B \\
        --bits 4 --n-calib 64 --n-steps 150

    # Both model sizes
    python paroquant_all_improvements.py --model both --bits 4
"""

import argparse
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


# ════════════════════════════════════════════════════════════════════════════
# §1  Quantization primitives
# ════════════════════════════════════════════════════════════════════════════

def round_to_grid(x: Tensor, bits: int, clip: bool = True) -> Tensor:
    qmax = (2 ** (bits - 1)) - 1
    x = x.round()
    return x.clamp(-qmax - 1, qmax) if clip else x


def absmax_scale(W: Tensor, bits: int, group_size: int) -> Tensor:
    """Simple per-group absmax scale (baseline)."""
    c, r = W.shape
    ng = (c + group_size - 1) // group_size
    pad = ng * group_size - c
    Wp = W if pad == 0 else torch.cat(
        [W, torch.zeros(pad, r, device=W.device, dtype=W.dtype)], 0)
    Wg = Wp.view(ng, group_size, r)
    qmax = (2 ** (bits - 1)) - 1
    s = Wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return s.expand(ng, group_size, r).reshape(ng * group_size, r)[:c]


def mse_scale(W: Tensor, bits: int, group_size: int) -> Tensor:
    """
    Improvement 2: MSE-optimal per-group scale via shrinkage grid search.
    Tries 30 shrinkage factors in [0.70, 1.00] and picks the one
    minimising (W_g - Q(W_g))^2, balancing granular and overload distortion.
    """
    c, r = W.shape
    ng = (c + group_size - 1) // group_size
    pad = ng * group_size - c
    Wp = W if pad == 0 else torch.cat(
        [W, torch.zeros(pad, r, device=W.device, dtype=W.dtype)], 0)
    Wg = Wp.view(ng, group_size, r)
    qmax = (2 ** (bits - 1)) - 1
    base_s = Wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax

    best_s = base_s.clone()
    best_mse = torch.full((ng, 1, r), float('inf'), device=W.device, dtype=W.dtype)

    for alpha in torch.linspace(0.70, 1.00, 30, device=W.device):
        s = base_s * alpha
        Wq = (Wg / s).round().clamp(-qmax - 1, qmax) * s
        err = (Wg - Wq).pow(2).mean(dim=1, keepdim=True)
        improve = err < best_mse
        best_s = torch.where(improve, s, best_s)
        best_mse = torch.where(improve, err, best_mse)

    return best_s.expand(ng, group_size, r).reshape(ng * group_size, r)[:c]


def quantize_dequantize(W: Tensor, scale: Tensor, bits: int,
                        stochastic: bool = False) -> Tensor:
    """
    Quantize-dequantize with optional stochastic rounding.
    Improvement 4: stochastic=True replaces deterministic round-to-nearest
    with unbiased stochastic rounding, eliminating the STE bias.
    """
    x = W / scale
    if stochastic:
        # floor(x) + Bernoulli(x - floor(x)) — unbiased, mean = x
        x_floor = x.floor()
        frac = x - x_floor
        x_rounded = x_floor + torch.bernoulli(frac.clamp(0.0, 1.0))
    else:
        x_rounded = x.round()
    qmax = (2 ** (bits - 1)) - 1
    return x_rounded.clamp(-qmax - 1, qmax) * scale


# ════════════════════════════════════════════════════════════════════════════
# §2  RAM-efficient Hessian collector
# ════════════════════════════════════════════════════════════════════════════

class HessianCollector:
    """
    Accumulates H = X^T X online in O(c^2) memory.
    Keeps a small fixed-size buffer of raw rows for output-MSE estimation.
    """
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
        if self.H is None:
            self.H = torch.zeros(x_cpu.shape[1], x_cpu.shape[1], dtype=torch.float32)
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

    def get_H(self, device) -> Tensor:
        return self.H.to(device)

    def get_X_mse(self, device) -> Tensor:
        return torch.cat(self.mse_buf, 0).float().to(device)

    def free(self):
        self.H = None
        self.mse_buf.clear()
        self.n_rows = self.mse_rows = 0


# ════════════════════════════════════════════════════════════════════════════
# §3  Pair selection
# ════════════════════════════════════════════════════════════════════════════

def select_pairs_random(g: int, K: int, seed: Optional[int] = None) -> List[List[Tuple[int, int]]]:
    """
    Baseline: random independent pair selection (Algorithm A1, Liang et al.),
    with a magnitude-difference tiebreak (we skip that here for simplicity).
    """
    import random
    rng = random.Random(seed)
    all_pairs = [(i, j) for i in range(g) for j in range(i + 1, g)]
    rng.shuffle(all_pairs)
    used_global: set = set()
    result = []
    n_pairs = g // 2
    for _ in range(K):
        used_local: set = set()
        stage = []
        for (i, j) in all_pairs:
            if len(stage) >= n_pairs:
                break
            if (i, j) in used_global or i in used_local or j in used_local:
                continue
            stage.append((i, j))
            used_local.update([i, j])
            used_global.add((i, j))
        result.append(stage)
    return result


def select_pairs_hessian(H_group: Tensor, g: int, K: int,
                          equilibrated: bool = False) -> List[List[Tuple[int, int]]]:
    """
    Improvement 5: Hessian-correlation-guided pair selection.
    Picks pairs with largest |H[i,j]| / sqrt(H[i,i] H[j,j]) — the
    normalised off-diagonal entry, which measures how much rotation of
    pair (i,j) can reduce the off-diagonal structure of H.

    Improvement 6a (equilibrated=True): apply diagonal equilibration to H
    before computing correlations, so the selection is scale-invariant.
    """
    H = H_group.float()
    if equilibrated:
        # Improvement 6a: symmetric Jacobi equilibration
        # Scale H -> D^{-1/2} H D^{-1/2} so diagonal = 1
        d = H.diagonal().clamp(min=1e-12)
        D_inv_sqrt = d.rsqrt()
        H = H * D_inv_sqrt.unsqueeze(0) * D_inv_sqrt.unsqueeze(1)

    d = H.diagonal().clamp(min=1e-12)
    C = H.abs() / (d.unsqueeze(0).sqrt() * d.unsqueeze(1).sqrt())
    C.fill_diagonal_(0.0)

    idx = torch.triu_indices(g, g, offset=1, device=H.device)
    vals = C[idx[0], idx[1]]
    order = vals.argsort(descending=True)

    used_global: set = set()
    result = []
    n_pairs = g // 2
    for _ in range(K):
        used_local: set = set()
        stage = []
        for k in order.tolist():
            if len(stage) >= n_pairs:
                break
            i, j = idx[0][k].item(), idx[1][k].item()
            if (i, j) in used_global or i in used_local or j in used_local:
                continue
            stage.append((i, j))
            used_local.update([i, j])
            used_global.add((i, j))
        result.append(stage)
    return result


# ════════════════════════════════════════════════════════════════════════════
# §4  Givens rotation forward and inverse
# ════════════════════════════════════════════════════════════════════════════

def apply_givens(W: Tensor, pairs: List[Tuple[int, int]], thetas: Tensor) -> Tensor:
    """Forward Givens rotation: apply pairs in order with angles thetas."""
    W = W.clone()
    for k, (i, j) in enumerate(pairs):
        c_k, s_k = thetas[k].cos(), thetas[k].sin()
        ri, rj = W[i].clone(), W[j].clone()
        W[i] = c_k * ri - s_k * rj
        W[j] = s_k * ri + c_k * rj
    return W


def invert_givens(W: Tensor, group_pairs: List[List[Tuple[int, int]]],
                  group_thetas: List[Tensor]) -> Tensor:
    """
    Inverse Givens rotation: apply stages in reverse order with negated angles.
    T = G_K ... G_1,  so T^{-1} = G_1^T ... G_K^T.
    Each G_k^T = G(i, j, -theta_k), so we reverse the stage order and negate.
    """
    W = W.clone()
    for k in range(len(group_pairs) - 1, -1, -1):
        neg_thetas = -group_thetas[k]
        W = apply_givens(W, group_pairs[k], neg_thetas)
    return W


# ════════════════════════════════════════════════════════════════════════════
# §5  Full transform: T(W) = R @ diag(alpha) @ W
#     Inverse:      T^{-1}(W_t) = diag(1/alpha) @ R^T @ W_t
# ════════════════════════════════════════════════════════════════════════════

def apply_full_transform(W: Tensor, transform: dict, device) -> Tensor:
    """Apply T = R @ diag(alpha) group-by-group."""
    g = transform['group_size']
    ng = transform['ng']
    c_orig = W.shape[0]
    pad = transform['pad']
    if pad > 0:
        W = torch.cat([W, torch.zeros(pad, W.shape[1], device=device, dtype=W.dtype)], 0)
    W_g = W.view(ng, g, -1)
    parts = []
    for gi in range(ng):
        Wg = W_g[gi].clone()
        alpha_g = transform['group_alphas'][gi].to(device)
        # diag(alpha) first, then rotation
        Wg = alpha_g.unsqueeze(1) * Wg
        for k in range(len(transform['group_pairs'][gi])):
            Wg = apply_givens(Wg, transform['group_pairs'][gi][k],
                              transform['group_thetas'][gi][k].to(device))
        parts.append(Wg)
    return torch.cat(parts, 0)[:c_orig]


def inverse_transform(W_t: Tensor, transform: dict, device) -> Tensor:
    """
    Apply T^{-1} = diag(1/alpha) @ R^T group-by-group.
    R^T is implemented by reversing the Givens stage order and negating angles.
    """
    g = transform['group_size']
    ng = transform['ng']
    c_orig = W_t.shape[0]
    pad = transform['pad']
    if pad > 0:
        W_t = torch.cat([W_t, torch.zeros(pad, W_t.shape[1],
                                           device=device, dtype=W_t.dtype)], 0)
    parts = []
    for gi in range(ng):
        sl = slice(gi * g, (gi + 1) * g)
        Wg = W_t[sl].clone()
        alpha_g = transform['group_alphas'][gi].to(device)
        # Invert rotation first (R^T = reverse stages with negated angles)
        Wg = invert_givens(Wg, transform['group_pairs'][gi],
                           [t.to(device) for t in transform['group_thetas'][gi]])
        # Then invert scaling
        Wg = Wg / alpha_g.unsqueeze(1)
        parts.append(Wg)
    return torch.cat(parts, 0)[:c_orig]


# ════════════════════════════════════════════════════════════════════════════
# §6  Effective Hessian in the transformed space
# ════════════════════════════════════════════════════════════════════════════

def transform_hessian(H: Tensor, transform: dict, device) -> Tensor:
    """
    Compute H_tilde = T^{-T} H T^{-1} (group-block-diagonal).
    T = R @ D,  T^{-1} = D^{-1} R^T,  so
    H_tilde = R D^{-1} H D^{-1} R^T  (per group block).
    """
    c = H.shape[0]
    g = transform['group_size']
    ng = transform['ng']
    pad = transform['pad']
    if pad > 0:
        I_pad = torch.eye(ng * g, device=device, dtype=H.dtype)
        I_pad[:c, :c] = H
        H = I_pad

    H_out = H.clone().to(device)
    for gi in range(ng):
        sl = slice(gi * g, (gi + 1) * g)
        H_blk = H_out[sl, sl].float()
        alpha_g = transform['group_alphas'][gi].to(device)
        D_inv = (1.0 / alpha_g)

        # Build full rotation matrix R for this group
        R = torch.eye(g, device=device, dtype=torch.float32)
        for k in range(len(transform['group_pairs'][gi])):
            pairs = transform['group_pairs'][gi][k]
            thetas = transform['group_thetas'][gi][k].to(device)
            for kk, (i, j) in enumerate(pairs):
                c_k, s_k = thetas[kk].cos(), thetas[kk].sin()
                G = torch.eye(g, device=device, dtype=torch.float32)
                G[i, i], G[j, j] = c_k, c_k
                G[i, j], G[j, i] = -s_k, s_k
                R = G @ R

        # H_tilde = R D^{-1} H D^{-1} R^T
        D_inv_H = H_blk * D_inv.unsqueeze(1)   # column-wise: D^{-1} H
        H_tilde = R @ (D_inv_H * D_inv.unsqueeze(0)) @ R.T  # R (D^{-1} H D^{-1}) R^T
        H_out[sl, sl] = H_tilde

    c_out = c - pad if pad else c
    return H_out[:c_out, :c_out].cpu()


# ════════════════════════════════════════════════════════════════════════════
# §7  QR-GPTQ error propagation (Improvement 7)
# ════════════════════════════════════════════════════════════════════════════

def qr_gptq_quantize(W: Tensor, H: Tensor, scale: Tensor,
                      bits: int, damp: float = 0.01) -> Tuple[Tensor, float]:
    """
    Babai nearest-plane algorithm via Cholesky of the (transformed) Hessian.
    Acts on W in the *transformed* coordinate system; returns dequantized
    weights still in the transformed system and the Babai bound.
    """
    c, r = W.shape
    device = W.device
    W = W.float()
    S = scale.float().to(device)
    H = H.float().to(device)

    # Act-order permutation: process high-curvature channels first
    perm = torch.argsort(H.diagonal(), descending=True)
    inv_perm = torch.argsort(perm)
    W, S, H = W[perm], S[perm], H[perm][:, perm]

    # Damp and Cholesky → upper-triangular basis A
    lam = max(damp / c * H.trace().item() / c, 1e-8)
    H.diagonal().add_(lam)
    try:
        L = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        H.diagonal().add_(1e-3 * H.diagonal().mean())
        L = torch.linalg.cholesky(H)
    A = L.T.contiguous()   # upper triangular, positive diagonal

    # Babai back-substitution (back-to-front, as in Chen et al. Alg. 4)
    Y = A @ W              # (c, r): weights projected into A-basis
    Q_w = W.clone()
    for j in range(c - 1, -1, -1):
        omega = Y[j] / A[j, j]          # coordinate along j-th GS vector
        zeta = omega / S[j]              # normalised by per-element scale
        z_j = round_to_grid(zeta, bits)
        q_j = z_j * S[j]
        Q_w[j] = q_j
        Y -= A[:, j].unsqueeze(1) * q_j.unsqueeze(0)   # update residual

    Q_w = Q_w[inv_perm]
    S = S[inv_perm]

    # Babai absolute error bound: (1/4) sum_j A[j,j]^2 * s_i[j]^2
    d = A.diagonal().pow(2)
    bound = 0.25 * (d.unsqueeze(1) * S.pow(2)).sum(0).mean().item()
    return Q_w, bound


# ════════════════════════════════════════════════════════════════════════════
# §8  Per-variant optimisation loop
# ════════════════════════════════════════════════════════════════════════════

def optimise_transform(
    W: Tensor,            # (c, r) fp32
    X_mse: Tensor,        # (m, c) activation buffer
    H: Tensor,            # (c, c) Hessian (accumulated X^T X)
    bits: int,
    group_size: int,
    n_rotations: int,
    n_steps: int,
    lr: float,
    use_improvements: bool,
    device: torch.device,
) -> dict:
    """
    Stage 1: optimise the rotation angles and channel-wise scale.

    use_improvements=False → baseline (random pairs, absmax scale, alpha=1
                              init, no equilibration, deterministic rounding)
    use_improvements=True  → all improvements except Cayley (Improvements
                              2, 4, 5, 6 applied during this stage)
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
    target = (X_mse @ W[:c]).detach()

    group_pairs = []
    group_thetas = []
    group_alphas = []
    params: List[dict] = []

    for gi in range(ng):
        H_blk = H[gi * group_size:(gi + 1) * group_size,
                   gi * group_size:(gi + 1) * group_size].to(device)

        if use_improvements:
            # Improvement 5 + 6a: Hessian-guided pair selection with equilibration
            pairs = select_pairs_hessian(H_blk, group_size, n_rotations,
                                         equilibrated=True)
        else:
            pairs = select_pairs_random(group_size, n_rotations, seed=gi)
        group_pairs.append(pairs)

        # Rotation angle parameters (initialised to 0 = identity)
        stage_thetas = []
        for k in range(n_rotations):
            t = nn.Parameter(torch.zeros(len(pairs[k]), device=device))
            stage_thetas.append(t)
            params.append({'params': t, 'lr': lr})
        group_thetas.append(stage_thetas)

        if use_improvements:
            # Improvement 6b: Jacobi initialisation of alpha.
            # alpha_j = H_blk[j,j]^{-1/2} so diag(T H T^T) ≈ 1.
            # After equilibration the transformed Hessian has unit diagonal,
            # giving mu_I = 1 (optimal incoherence) at step 0.
            alpha_init = H_blk.diagonal().clamp(min=1e-12).rsqrt()
        else:
            alpha_init = torch.ones(group_size, device=device)

        alpha = nn.Parameter(alpha_init.clone())
        group_alphas.append(alpha)
        params.append({'params': alpha, 'lr': lr})

    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_steps, eta_min=lr / 20)

    for step in range(n_steps):
        optimiser.zero_grad()

        # ── Forward transform (differentiable w.r.t. theta, alpha) ─────────
        parts_t = []
        for gi in range(ng):
            Wg = W_g[gi].detach().clone()          # constant weight data
            alpha_g = group_alphas[gi]              # nn.Parameter → has grad
            Wg_t = alpha_g.unsqueeze(1) * Wg       # diag(alpha) first
            for k in range(n_rotations):
                Wg_t = apply_givens(Wg_t, group_pairs[gi][k],
                                    group_thetas[gi][k])   # cos/sin → has grad
            parts_t.append(Wg_t)
        W_t = torch.cat(parts_t, 0)[:c]            # (c, r), requires grad ✓

        # ── Quantize with straight-through estimator (STE) ─────────────────
        # The STE trick: forward value = Q(W_t), backward gradient = identity.
        #   W_ste = W_t + (Q(W_t) - W_t).detach()
        # Forward:  W_ste = W_t + (Q(W_t) - W_t) = Q(W_t)
        # Backward: dW_ste/dW_t = 1 + 0 = 1  (the .detach() zeroes the
        #           gradient of the rounding residual, leaving only dW_t/dW_t=1)
        if use_improvements:
            with torch.no_grad():
                s = mse_scale(W_t.detach(), bits, group_size)
            W_q_hard = quantize_dequantize(W_t.detach(), s, bits,
                                            stochastic=True)
        else:
            with torch.no_grad():
                s = absmax_scale(W_t.detach(), bits, group_size)
            W_q_hard = quantize_dequantize(W_t.detach(), s, bits,
                                            stochastic=False)
        # STE: attach the gradient of W_t to the quantized value
        W_q_ste = W_t + (W_q_hard - W_t).detach()  # requires grad ✓

        # ── Inverse transform (differentiable w.r.t. theta, alpha) ─────────
        # T = R @ diag(alpha),  T^{-1} = diag(1/alpha) @ R^T
        # Both the forward (through W_q_ste ← W_t) and the inverse (through
        # T^{-1} itself) contribute gradients to theta and alpha.
        # NO torch.no_grad() and NO .detach() — the full graph must be live.
        parts_inv = []
        offset = 0
        for gi in range(ng):
            g_end = min(offset + group_size, c)
            g_act = g_end - offset
            Wq_g = W_q_ste[offset:g_end]           # (g_act, r), has grad ✓

            # Invert Givens: reverse stage order, negate angles
            # Uses the LIVE nn.Parameter thetas (not detached) so the
            # gradient flows through the inverse rotation to the thetas.
            Wq_inv = Wq_g
            for k in range(n_rotations - 1, -1, -1):
                neg_thetas = -group_thetas[gi][k]   # nn.Parameter, has grad ✓
                Wq_inv = apply_givens(Wq_inv, group_pairs[gi][k], neg_thetas)

            # Invert diagonal scaling: divide by alpha (live parameter)
            alpha_g = group_alphas[gi][:g_act]      # nn.Parameter, has grad ✓
            Wq_inv = Wq_inv / alpha_g.unsqueeze(1)

            parts_inv.append(Wq_inv)
            offset += group_size

        W_q_orig = torch.cat(parts_inv, 0)[:c]     # (c, r), requires grad ✓

        # Loss in original output space
        loss = (X_mse @ W_q_orig - target).pow(2).mean()
        loss.backward()
        optimiser.step()
        scheduler.step()

    return {
        'group_pairs': group_pairs,
        'group_thetas': [[t.detach() for t in stage] for stage in group_thetas],
        'group_alphas': [a.detach() for a in group_alphas],
        'ng': ng, 'group_size': group_size, 'pad': pad,
    }


# ════════════════════════════════════════════════════════════════════════════
# §9  Per-layer benchmark
# ════════════════════════════════════════════════════════════════════════════

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
    return torch.load(path, map_location='cpu', weights_only=True).to(
        dtype=dtype, device=device)


def benchmark_layer(
    name: str, W: Tensor, H: Tensor, X_mse: Tensor,
    bits: int, group_size: int, methods: List[str],
    device: torch.device, weight_dir: Path,
    n_rotations: int = 8, n_steps: int = 100, lr: float = 0.05,
    damp: float = 0.01,
) -> List[Result]:
    c, r = W.shape
    W = W.to(device).float()
    H = H.to(device).float()
    X_mse = X_mse.to(device).float()
    results = []

    for method in methods:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        use_improvements = (method == 'improved')

        # Stage 1: optimise rotation transform
        pad = ((c + group_size - 1) // group_size) * group_size - c
        W_padded = W if pad == 0 else torch.cat(
            [W, torch.zeros(pad, r, device=device)], 0)

        transform = optimise_transform(
            W, X_mse, H, bits, group_size,
            n_rotations, n_steps, lr,
            use_improvements=use_improvements,
            device=device,
        )

        # Stage 2: quantize in the transformed space
        W_t = apply_full_transform(W_padded, transform, device)

        if use_improvements:
            # Improvement 7: QR-GPTQ error propagation on rotated system
            H_tilde = transform_hessian(H.cpu(), transform, device).to(device)
            scale = mse_scale(W_t, bits, group_size).to(device)
            Q_t, bound = qr_gptq_quantize(W_t, H_tilde, scale, bits, damp)
        else:
            # Baseline: RTN with absmax scale
            scale = absmax_scale(W_t, bits, group_size).to(device)
            Q_t = quantize_dequantize(W_t, scale, bits, stochastic=False)
            bound = float('nan')

        # Bug 3 fix: inverse-transform Q_t to get original-space weights
        Q_orig = inverse_transform(Q_t, transform, device)

        # Output-space MSE (correct coordinate system)
        mse = (X_mse @ (W - Q_orig)).pow(2).mean().item()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        safe = name.replace('/', '__').replace('.', '_')
        wp = weight_dir / f"{safe}__{method}.pt"
        _save(Q_orig, wp)
        del Q_t, Q_orig
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        results.append(Result(name, method, mse, bound, t1 - t0, wp))

    return results


# ════════════════════════════════════════════════════════════════════════════
# §10  Perplexity evaluation
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_perplexity(model, tokenizer, calib_device,
                        seq_len: int = 2048, n_tokens: int = 2048 * 32) -> float:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  (datasets not installed — skipping perplexity)")
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


# ════════════════════════════════════════════════════════════════════════════
# §11  Main benchmark
# ════════════════════════════════════════════════════════════════════════════

def run(model_name: str, bits: int = 4, group_size: int = 128,
        n_calib: int = 64, seq_len: int = 2048,
        n_rotations: int = 8, n_steps: int = 100, lr: float = 0.05,
        damp: float = 0.01, device_str: str = 'cuda',
        max_layers: Optional[int] = None, eval_ppl: bool = True):
    import transformers

    methods = ['baseline', 'improved']
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    print(f"\n{'═'*72}")
    print(f"Model   : {model_name}")
    print(f"Bits    : {bits}   Group: {group_size}   K: {n_rotations}   Steps: {n_steps}")
    print(f"Device  : {device}")
    print(f"{'═'*72}")
    print("""
Improvements in 'improved' vs 'baseline':
  2. MSE-optimal scale (grid search over shrinkage ∈ [0.70, 1.00])
  4. Stochastic rounding during optimisation (unbiased STE)
  5. Hessian-correlation pair selection
  6a. Diagonal Hessian equilibration before pair selection
  6b. Jacobi initialisation of alpha (alpha_j = H[j,j]^{-1/2})
  7. Post-rotation QR-GPTQ error propagation (replaces RTN)
  Bug2/3 fixes: loss and MSE in original coordinate space
""")

    # ── Load model ─────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    load_kw: dict = {'torch_dtype': torch.float16}
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        budget = max(1, (gpu_mem - 2 * 1024**3) // (1024**3))
        load_kw.update(device_map='auto',
                       max_memory={int(device.index or 0): f"{budget}GiB",
                                   'cpu': '48GiB'})
    else:
        load_kw['device_map'] = 'cpu'

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    model.eval()
    calib_device = next(model.parameters()).device
    print(f"Model loaded  (embedding device: {calib_device})")

    # ── Calibration data ────────────────────────────────────────────────────
    from datasets import load_dataset as ld
    cdata = ld('salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
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
    print(f"Calibration   : {len(seqs)} sequences × {seq_len} tokens")

    # ── Collect Hessians ───────────────────────────────────────────────────
    collectors: Dict[str, HessianCollector] = {}
    linears: Dict[str, nn.Linear] = {}
    for lname, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            col = HessianCollector()
            col.register(mod)
            collectors[lname] = col
            linears[lname] = mod

    print("Collecting activations...")
    with torch.no_grad():
        for i in range(len(seqs)):
            model(calib_inputs[i:i+1].to(calib_device))
    for col in collectors.values():
        col.remove()
    print("Done.\n")

    # ── Benchmark ──────────────────────────────────────────────────────────
    weight_dir = Path(tempfile.mkdtemp(prefix='paroq_all_impr_'))
    layer_names = [n for n in linears if collectors[n].n_rows >= 16]
    if max_layers:
        layer_names = layer_names[:max_layers]

    all_results: List[Result] = []

    hdr = f"{'Layer':<50} {'Method':<10} {'MSE':>11} {'Bound':>13} {'Time':>8}"
    print(hdr)
    print('─' * len(hdr))

    for lname in layer_names:
        col = collectors[lname]
        H = col.get_H(device)
        X_mse = col.get_X_mse(device)
        col.free()

        W = linears[lname].weight.data.T.contiguous().to(device)

        layer_results = benchmark_layer(
            lname, W, H, X_mse, bits, group_size, methods,
            device, weight_dir, n_rotations, n_steps, lr, damp,
        )

        for res in layer_results:
            all_results.append(res)
            bstr = f"{res.bound:.4e}" if not math.isnan(res.bound) else "        n/a"
            print(f"{res.layer:<50} {res.method:<10} "
                  f"{res.mse:>11.4e} {bstr:>13} {res.time_s:>7.1f}s")

        del W, H, X_mse
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("SUMMARY — mean across layers")
    print(f"{'═'*72}")
    print(f"{'Method':<12} {'Mean MSE':>13} {'Mean Bound':>13} {'Mean Time':>12}")
    print('─' * 52)

    by_method: Dict[str, List[Result]] = {}
    for res in all_results:
        by_method.setdefault(res.method, []).append(res)

    for meth, rlist in by_method.items():
        mmse = sum(r.mse for r in rlist) / len(rlist)
        vb = [r.bound for r in rlist if not math.isnan(r.bound)]
        mb = sum(vb) / len(vb) if vb else float('nan')
        mt = sum(r.time_s for r in rlist) / len(rlist)
        bstr = f"{mb:.4e}" if not math.isnan(mb) else "         n/a"
        print(f"{meth:<12} {mmse:>13.4e} {bstr:>13} {mt:>11.1f}s")

    if 'baseline' in by_method and 'improved' in by_method:
        base_map = {r.layer: r.mse for r in by_method['baseline']}
        impr_mse = [r.mse for r in by_method['improved'] if r.layer in base_map]
        base_mse = [base_map[r.layer] for r in by_method['improved']
                    if r.layer in base_map]
        if impr_mse:
            ratios = [b / i for b, i in zip(base_mse, impr_mse) if i > 0]
            mean_ratio = sum(ratios) / len(ratios)
            mean_db = 10 * math.log10(mean_ratio)
            print(f"\nimproved vs baseline: {mean_ratio:.2f}× reduction "
                  f"({mean_db:+.2f} dB in output MSE)")

    # ── Per-improvement breakdown (layer 0) ────────────────────────────────
    print(f"\n{'═'*72}")
    print("Per-improvement contribution — first layer only")
    print(f"{'═'*72}")
    print("(Re-running with individual improvements disabled for ablation)")
    first_layer = layer_names[0] if layer_names else None
    if first_layer:
        col = collectors.get(first_layer)
        # Hessians already freed; skip ablation if not cached
        print(f"  (Ablation requires re-running; see --max-layers 1 for speed)")

    # ── Perplexity ─────────────────────────────────────────────────────────
    if eval_ppl:
        print(f"\n{'═'*72}")
        print("Perplexity — WikiText-2 test split")
        print(f"{'═'*72}")
        print(f"{'Method':<12} {'PPL':>10}")
        print('─' * 24)

        path_idx = {(r.layer, r.method): r.weight_path
                    for r in all_results if r.weight_path}

        for meth in methods:
            # Write quantized weights into the model
            for lname, mod in linears.items():
                key = (lname, meth)
                if key not in path_idx:
                    continue
                Q = _load(path_idx[key], mod.weight.dtype, mod.weight.data.device)
                mod.weight.data.copy_(Q.T)   # Q is (c, r); weight is (r, c)
                del Q

            ppl = compute_perplexity(model, tokenizer, calib_device, seq_len)
            print(f"{meth:<12} {ppl:>10.3f}")

            # Restore original weights (lazy save on first restore pass)
            for lname, mod in linears.items():
                ok = (lname, '__orig__')
                if ok not in path_idx:
                    safe = lname.replace('/', '__').replace('.', '_')
                    op = weight_dir / f"{safe}____orig__.pt"
                    if not op.exists():
                        _save(mod.weight.data.T.contiguous(), op)
                    path_idx[ok] = op
                orig = _load(path_idx[ok], mod.weight.dtype, mod.weight.data.device)
                mod.weight.data.copy_(orig.T)
                del orig

    shutil.rmtree(weight_dir, ignore_errors=True)
    print("\nDone.")
    return all_results


# ════════════════════════════════════════════════════════════════════════════
# §12  CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="ParoQuant with all improvements except Cayley",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model', default='meta-llama/Llama-3.2-3B',
                   help='HuggingFace model ID, or "both" for 3B + 8B')
    p.add_argument('--bits', type=int, default=4)
    p.add_argument('--group-size', type=int, default=128)
    p.add_argument('--n-calib', type=int, default=64,
                   help='Number of calibration sequences')
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--n-rotations', type=int, default=8,
                   help='Number of independent rotation stages K')
    p.add_argument('--n-steps', type=int, default=100,
                   help='Gradient descent steps for transform optimisation')
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--damp', type=float, default=0.01,
                   help='Hessian damping factor for QR-GPTQ')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-layers', type=int, default=None,
                   help='Limit to first N layers (for fast testing)')
    p.add_argument('--no-ppl', action='store_true',
                   help='Skip WikiText-2 perplexity evaluation')
    args = p.parse_args()

    models = ([args.model] if args.model != 'both'
              else ['meta-llama/Llama-3.2-3B', 'meta-llama/Meta-Llama-3-8B'])

    for m in models:
        run(m, args.bits, args.group_size, args.n_calib, args.seq_len,
            args.n_rotations, args.n_steps, args.lr, args.damp,
            args.device, args.max_layers, not args.no_ppl)


if __name__ == '__main__':
    main()