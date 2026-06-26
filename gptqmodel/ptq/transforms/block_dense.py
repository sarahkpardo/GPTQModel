# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Block-diagonal dense transforms on the input feature axis.

Weight layouts:

- ``linear``: ``nn.Linear`` weight ``(out_features, in_features)``
- ``conv1d``: ``transformers.Conv1D`` weight ``(in_features, out_features)``

Conventions (input-axis blocks of size ``G``):

- Offline weight transform with ``T_W = Q``:
  - linear: ``W[:, sl] = W[:, sl] @ Q.T``
  - conv1d: ``W[sl, :] = Q @ W[sl, :]``
- Online activation transform with stored ``T_X = Q.T`` (shape ``(G, G, C)``):
  ``x[:, sl] = x[:, sl] @ T_X[:, :, i]``  (equivalently ``x @ Q.T`` per block)
- Hessian congruence (orthogonal ``Q``): ``H' = Q H Q^T`` block-wise, including cross blocks
- Bilinear constraint: ``T_X @ T_W = I``
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch

from ..inference_data import InferenceTransformData


def pad_columns(num_columns: int, block_size: int) -> Tuple[int, int, int]:
    """Return ``(padded_columns, pad, num_blocks)`` for input feature count."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    pad = (block_size - (num_columns % block_size)) % block_size
    padded = num_columns + pad
    num_blocks = padded // block_size
    return padded, pad, num_blocks


def random_orthogonal_block(
    block_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """QR-based random orthogonal ``Q`` with deterministic sign fix."""
    matrix = torch.randn(
        block_size,
        block_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def generate_random_orthogonal_blocks(
    block_size: int,
    num_blocks: int,
    *,
    seed: int,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float32,
    inference_dtype: torch.dtype = torch.float16,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return per-block ``T_W`` list and stacked ``T_X`` tensor ``(G, G, C)``."""
    if num_blocks <= 0:
        empty = torch.empty(
            block_size,
            block_size,
            0,
            device=device,
            dtype=inference_dtype,
        )
        return [], empty

    # QR sampling is always CPU-seeded for cross-device reproducibility; blocks are
    # moved to the target device only when applied to weights/Hessian.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    cpu_device = torch.device("cpu")
    t_w_blocks: List[torch.Tensor] = []
    t_x_blocks: List[torch.Tensor] = []
    for _ in range(num_blocks):
        q = random_orthogonal_block(
            block_size,
            generator=generator,
            device=cpu_device,
            dtype=compute_dtype,
        )
        t_w_blocks.append(q.detach())
        t_x_blocks.append(q.T.to(dtype=inference_dtype).detach())

    t_x_matrices = torch.stack(t_x_blocks, dim=2)
    return t_w_blocks, t_x_matrices


def _resolve_blocks(
    t_w_blocks: Sequence[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    return [block.to(device=device, dtype=dtype) for block in t_w_blocks]


def resolve_weight_layout(weight: torch.Tensor, input_columns: int) -> str:
    """Return ``linear`` or ``conv1d`` based on which axis matches calibration columns."""
    if input_columns <= 0:
        raise ValueError(f"input_columns must be positive, got {input_columns}.")
    if weight.shape[1] == input_columns:
        return "linear"
    if weight.shape[0] == input_columns:
        return "conv1d"
    raise ValueError(
        f"Weight shape {tuple(weight.shape)} is incompatible with input_columns={input_columns}."
    )


def apply_block_transform_to_weight(
    weight: torch.Tensor,
    t_w_blocks: Sequence[torch.Tensor],
    *,
    block_size: int,
    pad: int,
    input_columns: int | None = None,
    weight_layout: str | None = None,
) -> torch.Tensor:
    """Apply offline ``T_W`` along the input feature axis."""
    columns = int(input_columns if input_columns is not None else weight.shape[1])
    layout = weight_layout or resolve_weight_layout(weight, columns)
    padded, _, num_blocks = pad_columns(columns, block_size)
    device = weight.device
    dtype = weight.dtype
    blocks = _resolve_blocks(t_w_blocks, device, dtype)

    if layout == "linear":
        out_features, in_features = weight.shape
        if pad > 0:
            weight_work = torch.cat(
                [
                    weight,
                    torch.zeros(out_features, pad, device=device, dtype=dtype),
                ],
                dim=1,
            )
        else:
            weight_work = weight
        result = weight_work.clone()
        for block_idx in range(num_blocks):
            sl = slice(block_idx * block_size, (block_idx + 1) * block_size)
            q = blocks[block_idx]
            result[:, sl] = result[:, sl] @ q.T
        return result[:, :in_features]

    if layout == "conv1d":
        in_features, out_features = weight.shape
        if pad > 0:
            weight_work = torch.cat(
                [
                    weight,
                    torch.zeros(pad, out_features, device=device, dtype=dtype),
                ],
                dim=0,
            )
        else:
            weight_work = weight
        result = weight_work.clone()
        for block_idx in range(num_blocks):
            sl = slice(block_idx * block_size, (block_idx + 1) * block_size)
            q = blocks[block_idx]
            result[sl, :] = q @ result[sl, :]
        return result[:in_features, :]

    raise ValueError(f"Unsupported weight_layout `{layout}`.")


def apply_block_transform_to_hessian(
    hessian: torch.Tensor,
    t_w_blocks: Sequence[torch.Tensor],
    *,
    block_size: int,
    pad: int,
) -> torch.Tensor:
    """Congruence-transform ``H`` with block-diagonal orthogonal ``Q``.

    Uses the full congruence ``H' = Q H Q^T``, including cross blocks
    ``H'[i, j] = Q_i H[i, j] Q_j^T``.
    """
    num_columns = hessian.shape[0]
    padded, _, num_blocks = pad_columns(num_columns, block_size)
    if pad > 0:
        h_work = hessian.new_zeros(padded, padded)
        h_work[:num_columns, :num_columns] = hessian
    else:
        h_work = hessian.clone()

    device = hessian.device
    blocks = _resolve_blocks(t_w_blocks, device, torch.float32)
    h_float = h_work.float()
    h_out = h_work.new_zeros(h_work.shape)
    for row_idx in range(num_blocks):
        row_sl = slice(row_idx * block_size, (row_idx + 1) * block_size)
        q_row = blocks[row_idx]
        for col_idx in range(num_blocks):
            col_sl = slice(col_idx * block_size, (col_idx + 1) * block_size)
            q_col = blocks[col_idx]
            h_out[row_sl, col_sl] = (q_row @ h_float[row_sl, col_sl] @ q_col.T).to(
                dtype=h_work.dtype
            )

    return h_out[:num_columns, :num_columns]


def apply_block_transform_to_activation(
    activations: torch.Tensor,
    t_x_matrices: torch.Tensor,
    *,
    block_size: int,
    pad: int,
    original_columns: int | None = None,
) -> torch.Tensor:
    """Apply online ``T_X`` to activation tensor with last dim = in_features."""
    if activations.numel() == 0:
        return activations

    in_features = activations.shape[-1]
    orig_cols = original_columns if original_columns is not None else in_features
    padded, _, num_blocks = pad_columns(orig_cols, block_size)

    flat_shape = activations.shape
    x = activations.reshape(-1, in_features)
    if pad > 0 and x.shape[1] < padded:
        x = torch.cat(
            [
                x,
                torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype),
            ],
            dim=1,
        )

    x_work = x.clone()
    device = x.device
    dtype = x.dtype
    for block_idx in range(num_blocks):
        sl = slice(block_idx * block_size, (block_idx + 1) * block_size)
        t_x = t_x_matrices[:, :, block_idx].to(device=device, dtype=dtype)
        x_work[:, sl] = x_work[:, sl] @ t_x

    x_out = x_work[:, :orig_cols]
    return x_out.reshape(*flat_shape[:-1], orig_cols)


def build_inference_transform_data(
    t_x_matrices: torch.Tensor,
    *,
    block_size: int,
    precision: torch.dtype,
    pad: int = 0,
) -> InferenceTransformData:
    """Build dense ``(G, G, C)`` inference payload."""
    extra = {"pad": int(pad)} if pad else {}
    return InferenceTransformData(
        transform_type="dense",
        T_X_matrices=t_x_matrices.detach().cpu(),
        precision=precision,
        block_size=int(block_size),
        extra=extra,
    )


def _resolve_inference_pad(inference: InferenceTransformData, *, pad: int | None = None) -> int:
    if pad is not None:
        return int(pad)
    return int(inference.extra.get("pad", 0) or 0)


def build_dense_activation_hook(
    inference: InferenceTransformData,
    *,
    pad: int | None = None,
) -> Callable[..., None]:
    """Return a forward pre-hook applying dense block ``T_X`` to activations."""
    if inference.T_X_matrices is None:
        raise ValueError("Dense activation hook requires T_X_matrices.")

    t_x_matrices = inference.T_X_matrices
    block_size = int(inference.block_size)
    resolved_pad = _resolve_inference_pad(inference, pad=pad)

    def _hook(_module, args, kwargs):
        if not args and "input" not in kwargs:
            return None
        x = args[0] if args else kwargs.get("input")
        if not isinstance(x, torch.Tensor):
            return None
        original_columns = x.shape[-1]
        # Apply T_X in float32 using activation dtype for numerical stability; fp16
        # storage of Q.T can break the bilinear constraint enough to hurt PPL.
        transformed = apply_block_transform_to_activation(
            x.float(),
            t_x_matrices.to(device=x.device, dtype=torch.float32),
            block_size=block_size,
            pad=resolved_pad,
            original_columns=original_columns,
        ).to(dtype=x.dtype)
        if args:
            new_args = (transformed, *args[1:])
            return new_args, kwargs
        kwargs["input"] = transformed
        return args, kwargs

    return _hook
