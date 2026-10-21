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
    """Return ``value`` as a host NumPy array, converting a Torch tensor.

    Torch tensors are detached and copied to host memory. NumPy and other
    non-Torch inputs fail closed: the torch-first DR contract admits no NumPy
    payloads on the op or reset execution path, so this helper is the
    adapter-private torch-to-host conversion boundary, not a NumPy fallback.
    """
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    raise TypeError(
        "interval payload must be a torch.Tensor under the torch-first DR "
        f"contract, got {type(value).__name__}"
    )
