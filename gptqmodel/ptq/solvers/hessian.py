# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared Hessian inverse computation for PTQ transforms and quantizers."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ...quantization.config import QuantizeConfig
from ...quantization.npu_linalg import npu_inverse_cholesky_factor
from ...quantization.qr_gptq_linalg import cholesky_hessian_inverse, qr_hessian_inverse
from ...utils.logger import setup_logger

log = setup_logger()


@torch.inference_mode()
def compute_hessian_inverse(
    H: torch.Tensor,
    *,
    qcfg: QuantizeConfig,
    nsamples: int,
    qr_R: Optional[torch.Tensor] = None,
    module_name: str = "",
    uses_qr_factorization: Optional[bool] = None,
) -> Tuple[Optional[torch.Tensor], float]:
    """Return ``(H_inv, damp_percent)`` with damping / diagonal-floor retries."""
    diag_view = H.diagonal()
    orig_diag = diag_view.clone()

    if uses_qr_factorization is None:
        uses_qr_factorization = getattr(qcfg.hessian, "factorization", "cholesky") == "qr"

    use_qr = (
        qr_R is not None
        and nsamples > 0
        and uses_qr_factorization
        and H.device.type != "npu"
    )

    base_abs_max = torch.max(orig_diag.abs()).item()
    if not math.isfinite(base_abs_max) or base_abs_max == 0.0:
        base_abs_max = 1.0
    floor_base = base_abs_max * 1e-6
    max_floor_attempts = 6
    used_damp = qcfg.damp_percent
    last_error = None

    label = module_name or "<unknown>"
    attempt = 0
    while attempt <= max_floor_attempts:
        if attempt == 0:
            current_diag = orig_diag
        else:
            floor_increment = floor_base * math.pow(10.0, attempt - 1)
            current_diag = torch.clamp(orig_diag + floor_increment, min=floor_increment)
            if attempt == 1:
                log.warn(
                    "Quantization: Module `%s` -> Applying Hessian diagonal floor (+%.2e) "
                    "to recover positive definiteness.",
                    label,
                    floor_increment,
                )
            else:
                log.warn(
                    "Quantization: Module `%s` -> Increasing Hessian diagonal floor to +%.2e.",
                    label,
                    floor_increment,
                )

        diag_view.copy_(current_diag)
        mean = torch.mean(current_diag)
        damp = qcfg.damp_percent

        damp_recovery_started = False
        recovery_initial_damp = None
        recovery_last_damp = None

        while 0 < damp < 1:
            try:
                if use_qr:
                    hinv_result = qr_hessian_inverse(
                        qr_R,
                        nsamples=nsamples,
                        damp=damp,
                        damp_mean=float(mean.item()),
                    )
                else:
                    diag_view.add_(damp * mean)
                    if H.device.type == "npu":
                        hinv_result = npu_inverse_cholesky_factor(H)
                    else:
                        hinv_result = cholesky_hessian_inverse(H)
                    diag_view.copy_(current_diag)
                used_damp = damp
                if damp_recovery_started:
                    log.warn(
                        "Quantization: Module `%s` -> Damp recovery succeeded at "
                        "`damp_percent=%.5f` (started at %.5f).",
                        label,
                        damp,
                        recovery_initial_damp,
                    )
                return hinv_result, used_damp
            except torch._C._LinAlgError as exc:
                last_error = exc
                if not use_qr:
                    diag_view.copy_(current_diag)
                if qcfg.damp_auto_increment != 0:
                    if not damp_recovery_started:
                        damp_recovery_started = True
                        recovery_initial_damp = damp
                        log.warn(
                            "Quantization: Module `%s` -> Starting damp recovery at "
                            "`damp_percent=%.5f`, increment step `%.5f`.",
                            label,
                            damp,
                            qcfg.damp_auto_increment,
                        )
                    damp += qcfg.damp_auto_increment
                    recovery_last_damp = damp
                else:
                    factorization = "QR" if use_qr else "Cholesky"
                    log.warn(
                        "Quantization: Module `%s` -> Hessian %s failed with "
                        "`damp_percent=%.5f` and no auto increment configured.",
                        label,
                        factorization,
                        damp,
                    )
                    break

        if damp_recovery_started:
            final_damp = recovery_last_damp if recovery_last_damp is not None else damp
            log.warn(
                "Quantization: Module `%s` -> Damp recovery failed after reaching "
                "`damp_percent=%.5f`.",
                label,
                final_damp,
            )

        attempt += 1

    log.error(
        "Quantization: Module `%s` -> Hessian remained non positive-definite after "
        "diagonal floor attempts. Last `damp_percent` tried = %.5f.",
        label,
        damp,
    )
    if last_error is not None:
        log.debug("Hessian failure detail: %s", last_error)
    return None, 1.0
