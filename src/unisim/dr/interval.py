"""Declarative interval domain-randomization term descriptors.

An interval randomization plan is a tuple of :class:`IntervalTermOp` entries:
a term name, a host NumPy or device payload, and optional target body IDs.
The builtin specs below pin the payload contract for the terms shared between
UniLab's domain-randomization manager and the backend adapters. Custom terms are
free-form strings owned by a backend and validated only against that
backend's declared capability set; there is intentionally no mutable global
registry for them.

The module is stdlib + NumPy only and never imports an engine SDK. Host NumPy
ops and plans stay pickle-safe (protocol 4) across spawn-based collector
processes; device payloads are process-local by design and are validated by
the receiving in-process tensor adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

INTERVAL_TERM_PUSH = "push"
INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA = "body_linear_velocity_delta"
INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA = "body_angular_velocity_delta"
INTERVAL_TERM_BODY_FORCE = "body_force"
INTERVAL_TERM_BODY_TORQUE = "body_torque"

__all__ = [
    "INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA",
    "INTERVAL_TERM_BODY_FORCE",
    "INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA",
    "INTERVAL_TERM_BODY_TORQUE",
    "INTERVAL_TERM_PUSH",
    "INTERVAL_TERM_SPECS",
    "IntervalTermOp",
    "IntervalTermSpec",
    "TensorIntervalPayload",
    "interval_term_spec",
    "require_op_body_ids",
    "require_op_host_payload",
    "validate_interval_op",
]


@dataclass(frozen=True, slots=True)
class IntervalTermSpec:
    """Declared payload contract for one interval randomization term."""

    name: str
    requires_body_ids: bool
    payload_ndim: int


INTERVAL_TERM_SPECS: tuple[IntervalTermSpec, ...] = (
    IntervalTermSpec(
        name=INTERVAL_TERM_PUSH,
        requires_body_ids=False,
        payload_ndim=1,
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
        requires_body_ids=True,
        payload_ndim=3,
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
        requires_body_ids=True,
        payload_ndim=3,
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_FORCE,
        requires_body_ids=True,
        payload_ndim=3,
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_TORQUE,
        requires_body_ids=True,
        payload_ndim=3,
    ),
)


def interval_term_spec(name: str) -> IntervalTermSpec:
    """Return the builtin spec for ``name`` or raise a stable ``KeyError``."""
    for spec in INTERVAL_TERM_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown interval randomization term: {name!r}")


@dataclass(frozen=True)
class IntervalTermOp:
    """One interval randomization operation: term, payload, optional bodies.

    ``payload`` remains a NumPy array for host-resident plans. Device-resident
    producers package engine-native arrays in :class:`TensorIntervalPayload`
    and pass that object here; base validation deliberately stays Torch-free.
    """

    term: str
    payload: np.ndarray | TensorIntervalPayload
    body_ids: np.ndarray | None = None

    def validate(self) -> None:
        """Check the builtin spec contract; custom terms pass through."""
        validate_interval_op(self)


def require_op_body_ids(op: IntervalTermOp) -> np.ndarray:
    """Return a validated body-targeted op's ids, failing closed when absent.

    Builtin body terms are checked by :func:`validate_interval_op` before the
    backend handler runs; custom terms routed to a body handler fail closed
    here instead of surfacing an opaque ``None`` error deeper in the backend.
    """
    if op.body_ids is None:
        raise ValueError(f"interval term '{op.term}' requires body_ids")
    return op.body_ids


def require_op_host_payload(op: IntervalTermOp) -> np.ndarray:
    """Return one host NumPy interval payload, rejecting device payloads.

    Host adapters own this explicit conversion point. A device payload is not
    silently coerced or copied to NumPy; unsupported tensor data planes fail
    closed with an actionable type error.
    """
    if isinstance(op.payload, TensorIntervalPayload):
        raise TypeError(f"interval term '{op.term}' does not accept a device-resident payload")
    return op.payload


@dataclass(frozen=True, slots=True)
class TensorIntervalPayload:
    """Device-resident interval payload for direct tensor data planes.

    Fields are engine-native arrays (Torch tensors today) and are intentionally
    typed ``Any`` so the base contract does not import a tensor runtime. A
    device-resident producer owns generation and finiteness before handing the
    payload to an adapter. This initial contract carries the full-width
    world-frame push force table used by MJWarp; more fields can be added only
    with adapter capability declarations and regression coverage.
    """

    push_force: Any | None = None

    def requested_terms(self) -> frozenset[str]:
        return frozenset({INTERVAL_TERM_PUSH}) if self.push_force is not None else frozenset()

    def is_empty(self) -> bool:
        return self.push_force is None


def validate_interval_op(op: IntervalTermOp) -> None:
    """Validate ``op`` against its builtin spec, ignoring custom terms.

    Builtin terms must carry ``body_ids`` exactly when their spec requires
    them, and their payload must have the declared ndim.  A
    :class:`TensorIntervalPayload` defers structural checks to its
    device-resident adapter, which owns the accepted engine type, dtype,
    device, contiguity, and value rules. Unknown (custom) terms are
    backend-owned and pass through untouched.
    """
    try:
        spec = interval_term_spec(op.term)
    except KeyError:
        return
    if spec.requires_body_ids and op.body_ids is None:
        raise ValueError(f"interval term '{op.term}' requires body_ids")
    if not spec.requires_body_ids and op.body_ids is not None:
        raise ValueError(f"interval term '{op.term}' does not accept body_ids")
    payload = op.payload
    if isinstance(payload, TensorIntervalPayload):
        if payload.requested_terms() != frozenset({op.term}):
            raise ValueError(
                f"TensorIntervalPayload does not carry exactly the interval term {op.term!r}"
            )
        return
    payload_ndim = np.asarray(payload).ndim
    if payload_ndim != spec.payload_ndim:
        raise ValueError(
            f"interval term '{op.term}' payload must have ndim "
            f"{spec.payload_ndim}, got {payload_ndim}"
        )
