from __future__ import annotations

import numpy as np
import pytest

from unisim.utils.quaternion import (
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    torch_quat_mul,
    torch_quat_rotate,
    torch_quat_rotate_inverse,
)


def test_numpy_quaternion_primitives_compose_and_invert() -> None:
    angle = np.pi / 3
    quat = np.asarray((np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0))
    vector = np.asarray((1.0, 0.0, 0.0))
    np.testing.assert_allclose(quat_rotate(quat, vector), (0.5, 0.0, -np.sqrt(3) / 2), atol=1e-12)
    np.testing.assert_allclose(
        quat_rotate_inverse(quat, quat_rotate(quat, vector)), vector, atol=1e-12
    )
    np.testing.assert_allclose(
        quat_mul(quat, quat), (np.cos(angle), 0.0, np.sin(angle), 0.0), atol=1e-12
    )


def test_torch_quaternion_primitives_compose_and_invert() -> None:
    torch = pytest.importorskip("torch")
    angle = torch.tensor(np.pi / 4, dtype=torch.float64)
    quat = torch.stack(
        (torch.cos(angle / 2), torch.tensor(0.0), torch.sin(angle / 2), torch.tensor(0.0))
    )
    vector = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64)
    rotated = torch_quat_rotate(torch, quat, vector)
    assert torch.allclose(
        rotated, torch.tensor((np.sqrt(0.5), 0.0, -np.sqrt(0.5)), dtype=torch.float64)
    )
    assert torch.allclose(torch_quat_rotate_inverse(torch, quat, rotated), vector)
    np.testing.assert_allclose(
        torch_quat_mul(torch, quat, quat).numpy(), quat_mul(quat.numpy(), quat.numpy())
    )
