import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qr_gptq_linalg import (
    cholesky_hessian_inverse,
    merge_qr_factors,
    qr_hessian_inverse,
    update_qr_factor,
)


def test_incremental_qr_matches_single_batch_qr():
    torch.manual_seed(0)
    batches = [torch.randn(24, 6), torch.randn(17, 6), torch.randn(31, 6)]

    incremental = None
    for batch in batches:
        incremental = update_qr_factor(incremental, batch)

    stacked = torch.cat(batches, dim=0)
    _, reference = torch.linalg.qr(stacked, mode="reduced")

    lhs = incremental.T @ incremental
    rhs = reference.T @ reference
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_merge_qr_factors_matches_joint_gram():
    torch.manual_seed(1)
    X1 = torch.randn(20, 5)
    X2 = torch.randn(15, 5)
    R1 = update_qr_factor(None, X1)
    R2 = update_qr_factor(None, X2)
    merged = merge_qr_factors(R1, R2)

    joint = torch.cat([X1, X2], dim=0)
    expected = update_qr_factor(None, joint)
    assert torch.allclose(merged.T @ merged, expected.T @ expected, atol=1e-4, rtol=1e-4)


def test_qr_hessian_inverse_matches_cholesky_reference():
    torch.manual_seed(2)
    X = torch.randn(96, 8)
    nsamples = X.shape[0]
    damp = 0.05
    H = (2.0 / nsamples) * X.T @ X
    damp_mean = float(H.diagonal().mean().item())

    R = update_qr_factor(None, X)
    Hinv_qr = qr_hessian_inverse(
        R,
        nsamples=nsamples,
        damp=damp,
        damp_mean=damp_mean,
    )

    H_damped = H.clone()
    H_damped.diagonal().add_(damp * damp_mean)
    Hinv_chol = cholesky_hessian_inverse(H_damped)

    Hinv_true = torch.linalg.inv(H_damped)
    assert torch.allclose(Hinv_qr.transpose(-1, -2) @ Hinv_qr, Hinv_true, atol=1e-4, rtol=1e-3)
    assert torch.allclose(Hinv_chol.transpose(-1, -2) @ Hinv_chol, Hinv_true, atol=1e-4, rtol=1e-3)


def test_gptq_calibration_builds_qr_factor_and_inverse():
    module = nn.Linear(8, 4, bias=False)
    qcfg = QuantizeConfig(
        damp_percent=0.05,
        damp_auto_increment=0.05,
        hessian={"factorization": "qr"},
    )
    gptq = GPTQ(module, qcfg=qcfg)
    calib = torch.randn(32, 8)

    for _ in range(4):
        gptq.add_batch(calib.clone(), torch.empty(0))

    gptq.finalize_hessian()
    assert gptq._qr_R is not None
    assert gptq._qr_R.shape == (8, 8)

    Hinv, damp = gptq.hessian_inverse(gptq.H, qr_R=gptq._qr_R)
    assert Hinv is not None
    assert torch.allclose(Hinv, torch.triu(Hinv))

    damped = gptq.H.clone()
    damped.diagonal().add_(damp * damped.diagonal().mean())
    reconstructed = Hinv.transpose(-1, -2) @ Hinv
    assert torch.allclose(reconstructed, torch.linalg.inv(damped), atol=5e-4, rtol=5e-3)
