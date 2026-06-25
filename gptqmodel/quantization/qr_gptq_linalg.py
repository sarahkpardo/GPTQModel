# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""
QR-GPTQ linear algebra helpers.

Implements the numerically stable QR factorization path described in
Chen et al. (2025) and Birnick (2026): factor the augmented activation
matrix via Householder QR instead of forming the Gram matrix H = X^T X and
running Cholesky.  The upper-triangular factor A satisfies A^T A = H and
the GPTQ error-propagation matrix is H^{-1/2}_upper = A^{-T}.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def normalize_qr_diagonal(A: torch.Tensor) -> torch.Tensor:
    """Enforce a unique QR convention with a positive diagonal (display only)."""
    signs = A.diagonal().sign()
    signs[signs == 0] = 1.0
    return A * signs.unsqueeze(0)


def update_qr_factor(
    R: Optional[torch.Tensor],
    X_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Incrementally update the thin QR factor R so that R^T R = X_all^T X_all.

    Given an existing factor R from prior calibration batches and a new batch
    X_batch, the merged factor is obtained by QR([R; X_batch]).

    Sign normalization is intentionally omitted here so that the Gram identity
    R^T R = X^T X is preserved exactly.
    """
    X = X_batch.to(dtype=torch.float32)
    if X.shape[0] == 0:
        if R is None:
            raise ValueError("update_qr_factor: empty batch with no existing factor")
        return R

    if R is None:
        _, R_new = torch.linalg.qr(X, mode="reduced")
    else:
        stacked = torch.cat([R, X], dim=0)
        _, R_new = torch.linalg.qr(stacked, mode="reduced")

    return R_new


def merge_qr_factors(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Merge two partial QR factors accumulated on different devices."""
    stacked = torch.cat([R_a.to(torch.float32), R_b.to(torch.float32)], dim=0)
    _, R_merged = torch.linalg.qr(stacked, mode="reduced")
    return R_merged


def qr_upper_hessian_factor(
    R_raw: torch.Tensor,
    *,
    nsamples: int,
    damp: float,
    damp_mean: float,
) -> torch.Tensor:
    """
    Apply GPTQ damping and return the upper-triangular A with A^T A = H_damped.

    GPTQModel stores H = (2 / nsamples) * X^T X.  Damping adds
    ``damp * mean(diag(H))`` to each diagonal entry of H, equivalent to
    appending ``sqrt(damp * mean) * I`` rows to the scaled activation matrix.
    """
    if nsamples <= 0:
        raise ValueError("qr_upper_hessian_factor: nsamples must be positive")

    scale = math.sqrt(2.0 / float(nsamples))
    R_scaled = R_raw * scale
    sqrt_lam = math.sqrt(max(damp * damp_mean, 1e-8))
    damp_rows = sqrt_lam * torch.eye(
        R_raw.shape[0],
        device=R_raw.device,
        dtype=R_raw.dtype,
    )
    stacked = torch.cat([R_scaled, damp_rows], dim=0)
    _, A = torch.linalg.qr(stacked, mode="reduced")
    return A


def upper_inverse_cholesky_from_A(A: torch.Tensor) -> torch.Tensor:
    """
    Return Hinv, the upper-triangular Cholesky factor of H^{-1}, from QR factor A.

    When A^T A = H, we have H^{-1} = A^{-1} A^{-T} = U^T U with U = A^{-T}.
    """
    size = A.shape[0]
    identity = torch.eye(size, device=A.device, dtype=A.dtype)
    A_inv = torch.linalg.solve_triangular(A, identity, upper=True)
    return A_inv.T.contiguous()


def cholesky_hessian_inverse(H: torch.Tensor) -> torch.Tensor:
    """Classic GPTQ path: upper Cholesky factor of H^{-1}."""
    L = torch.linalg.cholesky(H)
    return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)


def qr_hessian_inverse(
    R_raw: torch.Tensor,
    *,
    nsamples: int,
    damp: float,
    damp_mean: float,
) -> torch.Tensor:
    """Compute the GPTQ error-propagation factor via QR of augmented activations."""
    A = qr_upper_hessian_factor(
        R_raw,
        nsamples=nsamples,
        damp=damp,
        damp_mean=damp_mean,
    )
    return upper_inverse_cholesky_from_A(A)


def qr_hessian_inverse_from_activations(
    X: torch.Tensor,
    *,
    damp: float,
    damp_mean: float,
    column_perm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    One-shot QR-GPTQ factor from a full activation matrix (benchmark / tests).

    X: (n, c) calibration activations in fp32.
    """
    X_fp = X.to(dtype=torch.float32)
    if column_perm is not None:
        X_fp = X_fp[:, column_perm]

    sqrt_lam = math.sqrt(max(damp * damp_mean, 1e-8))
    c = X_fp.shape[1]
    X_aug = torch.cat(
        [X_fp, sqrt_lam * torch.eye(c, device=X_fp.device, dtype=X_fp.dtype)],
        dim=0,
    )
    _, A = torch.linalg.qr(X_aug, mode="reduced")
    return upper_inverse_cholesky_from_A(A)


def apply_column_perm_to_qr(R: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute QR factor columns to match act-order reordering of H and W."""
    return R[:, perm.to(device=R.device)]


def babai_bound_from_A(A: torch.Tensor, scales: torch.Tensor) -> float:
    """Average-over-channels Babai absolute error bound from upper-triangular A."""
    d = A.diagonal().pow(2)
    per_channel = 0.25 * (d.unsqueeze(1) * scales.pow(2)).sum(dim=0)
    return per_channel.mean().item()
