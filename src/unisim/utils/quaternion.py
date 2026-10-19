"""Shared quaternion primitives for NumPy and optional Torch tensors."""

from __future__ import annotations

from typing import Any

import numpy as np


def quat_mul(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> np.ndarray:
    """Multiply broadcast-compatible wxyz quaternion arrays."""
    left = np.asarray(left_wxyz)
    right = np.asarray(right_wxyz)
    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError(
            f"expected trailing quaternion dimension 4, got {left.shape} and {right.shape}"
        )
    lead_shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = np.broadcast_to(left, (*lead_shape, 4))
    right = np.broadcast_to(right, (*lead_shape, 4))
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quat_rotate(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by broadcast-compatible wxyz quaternions."""
    quat = np.asarray(quat_wxyz)
    vector = np.asarray(vectors)
    if quat.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError(
            f"expected trailing dimensions 4 and 3, got {quat.shape} and {vector.shape}"
        )
    scalar = quat[..., 0:1]
    axis = quat[..., 1:4]
    lead_shape = np.broadcast_shapes(quat.shape[:-1], vector.shape[:-1])
    quat = np.broadcast_to(quat, (*lead_shape, 4))
    vector = np.broadcast_to(vector, (*lead_shape, 3))
    first = np.cross(axis, vector) * 2.0
    return vector + scalar * first + np.cross(axis, first)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors into wxyz-quaternion frames."""
    conjugate = np.asarray(quat_wxyz).copy()
    conjugate[..., 1:4] *= -1.0
    return quat_rotate(conjugate, vectors)


def torch_quat_mul(torch: Any, left_wxyz: Any, right_wxyz: Any) -> Any:
    """Multiply device-resident wxyz quaternions with Torch primitives."""
    left_w, left_xyz = left_wxyz[..., 0:1], left_wxyz[..., 1:4]
    right_w, right_xyz = right_wxyz[..., 0:1], right_wxyz[..., 1:4]
    if len(tuple(right_xyz.shape)) < len(tuple(left_xyz.shape)):
        right_xyz = right_xyz.expand_as(left_xyz)
    elif len(tuple(left_xyz.shape)) < len(tuple(right_xyz.shape)):
        left_xyz = left_xyz.expand_as(right_xyz)
    cross = torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    return torch.cat(
        (
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
            left_w * right_xyz + right_w * left_xyz + cross,
        ),
        dim=-1,
    )


def torch_quat_rotate(torch: Any, quat_wxyz: Any, vectors: Any) -> Any:
    """Rotate device-resident vectors by wxyz quaternions."""
    quat_vector = quat_wxyz[..., 1:4]
    if len(tuple(vectors.shape)) < len(tuple(quat_vector.shape)):
        vectors = vectors.expand_as(quat_vector)
    cross = torch.linalg.cross(quat_vector, vectors, dim=-1)
    second_cross = torch.linalg.cross(quat_vector, cross, dim=-1)
    return vectors + 2.0 * quat_wxyz[..., 0:1] * cross + 2.0 * second_cross


def torch_quat_rotate_inverse(torch: Any, quat_wxyz: Any, vectors: Any) -> Any:
    """Rotate device-resident vectors into wxyz-quaternion frames."""
    conjugate = torch.cat((quat_wxyz[..., 0:1], -quat_wxyz[..., 1:4]), dim=-1)
    return torch_quat_rotate(torch, conjugate, vectors)


def torch_cross(torch: Any, left: Any, right: Any) -> Any:
    """Evaluate a trailing-axis cross product on a Torch device."""
    return torch.linalg.cross(left, right, dim=-1)


__all__ = [
    "quat_mul",
    "quat_rotate",
    "quat_rotate_inverse",
    "torch_cross",
    "torch_quat_mul",
    "torch_quat_rotate",
    "torch_quat_rotate_inverse",
]
