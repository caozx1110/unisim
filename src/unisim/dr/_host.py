"""Adapter-private host conversions for torch-first DR payloads.

This module is deliberately private: it is not re-exported by ``unisim.dr``
and is not part of the public contract. NumPy materialization belongs at
adapter-owned host boundaries (host bridges, subprocess IPC workers, replay
leaves); the public payload family stays Torch tensors end to end.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from .types import TensorResetRandomizationPayload


def reset_payload_to_host_dict(
    payload: TensorResetRandomizationPayload,
) -> dict[str, np.ndarray]:
    """Materialize every populated reset field as a host NumPy array.

    Each Torch tensor is detached and copied to host memory
    (``tensor.detach().cpu().numpy()``); unpopulated fields are omitted.
    """
    if not isinstance(payload, TensorResetRandomizationPayload):
        raise TypeError("payload must be a TensorResetRandomizationPayload")
    return {
        payload_field.name: interval_payload_to_host(getattr(payload, payload_field.name))
        for payload_field in fields(payload)
        if getattr(payload, payload_field.name) is not None
    }


def interval_payload_to_host(value: Any) -> np.ndarray:
    """Return ``value`` as a host NumPy array, converting Torch tensors.

    NumPy inputs pass through unchanged (cold serialization/replay leaves).
    Torch tensors are detached and copied to host memory. Anything else fails
    closed with an actionable type error.
    """
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    raise TypeError(
        f"interval payload must be a torch.Tensor or NumPy array, got {type(value).__name__}"
    )
