"""Version-bound MJWarp continuation snapshots, with no optional-runtime imports.

Native Data is retained in full: re-running forward would change solver warmstart
and last-substep sensor state. Model randomization uses the adapter's declared
expanded field family. Immutable model/native layouts are bound by identity.
This is a local source implementation, not a claim of released UniSim support.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

import numpy as np

from .randomization import EXPANDED_MODEL_FIELDS, OPTION_VEC_FIELDS

SCHEMA = "unisim.mjwarp.continuation.v1"
_FLAGS = (
    "_tracked_body_state_dirty",
    "_tracked_body_host_dirty",
    "_host_cache_stale",
    "_xfrc_pending",
    "_device_xfrc_pending",
    "_completion_sequence",
)


def _require_supported(backend: Any) -> None:
    backend._require_entity_healthy()
    if backend._pre_step_control_fn is not None:
        raise NotImplementedError("MJWarp continuation does not support host control callbacks")
    if backend._fixed_variant_plan is not None:
        raise NotImplementedError("MJWarp continuation does not yet support fixed model variants")
    callback = backend._device_model.callback
    if any(value is not None for value in vars(callback).values()):
        raise NotImplementedError("MJWarp continuation does not support native physics callbacks")


def _array_leaves(value: Any, prefix: str, array_type: Any) -> dict[str, Any]:
    """Inventory arrays while rejecting unrecognized native state containers."""
    if isinstance(value, array_type):
        return {prefix: value}
    leaves: dict[str, Any] = {}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            leaves.update(
                _array_leaves(getattr(value, field.name), f"{prefix}.{field.name}", array_type)
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.update(_array_leaves(item, f"{prefix}.{index}", array_type))
    elif value is not None and not isinstance(value, (bool, int, float, str, np.generic)):
        raise TypeError(f"Unsupported native continuation field {prefix}: {type(value).__name__}")
    return leaves


def _arrays(backend: Any) -> dict[str, Any]:
    array_type = backend._warp.array
    result = _array_leaves(backend._device_data, "data", array_type)
    scratch = backend._reset_scratch_data
    if scratch is not None:
        result.update(_array_leaves(scratch, "reset_scratch", array_type))
    for name in EXPANDED_MODEL_FIELDS:
        result[f"model.{name}"] = getattr(backend._device_model, name)
    for name in OPTION_VEC_FIELDS:
        result[f"model.opt.{name}"] = getattr(backend._device_model.opt, name)
    # Fixed-address staging/caches include pending wrenches and reset masks. Host
    # mirrors can deliberately be stale; retain their bytes and stale flags together.
    for name, value in vars(backend).items():
        if isinstance(value, (np.ndarray, array_type)):
            result[f"adapter.{name}"] = value
    for index, view in enumerate(backend._physics_step_tensor_views):
        for name, value in vars(view).items():
            if isinstance(value, (np.ndarray, array_type)):
                result[f"substep.{index}.{name}"] = value
    return result


def _native_constants(value: Any, prefix: str, array_type: Any) -> dict[str, Any]:
    """Bind non-array Data fields as immutable layout/capacity configuration."""
    if isinstance(value, array_type):
        return {}
    result: dict[str, Any] = {}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            result.update(
                _native_constants(getattr(value, field.name), f"{prefix}.{field.name}", array_type)
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_native_constants(item, f"{prefix}.{index}", array_type))
    elif value is None or isinstance(value, (bool, int, float, str)):
        result[prefix] = value
    elif isinstance(value, np.generic):
        result[prefix] = value.item()
    else:
        raise TypeError(f"Unsupported native continuation constant: {prefix}")
    return result


def _identity(backend: Any) -> dict[str, Any]:
    # MJB includes compiled topology, options, names, defaults and asset contents.
    # Cache only this immutable constructor-owned CPU model fingerprint.
    fingerprint = getattr(backend, "_continuation_model_fingerprint", None)
    if fingerprint is None:
        buffer = np.empty(backend._mujoco.mj_sizeModel(backend._cpu_model), dtype=np.uint8)
        backend._mujoco.mj_saveModel(backend._cpu_model, buffer=buffer)
        fingerprint = hashlib.sha256(buffer).hexdigest()
        backend._continuation_model_fingerprint = fingerprint
    return {
        "model_sha256": fingerprint,
        "num_envs": backend._num_envs,
        "sim_dt": backend._sim_dt,
        "nconmax": backend._nconmax,
        "njmax": backend._njmax,
        "native_versions": {
            package: version(package) for package in ("mujoco", "mujoco-warp", "warp-lang")
        },
        "cuda_graph_enabled": backend._cuda_graph_enabled,
        "data_layout": _native_constants(backend._device_data, "data", backend._warp.array),
        "reset_scratch_layout": _native_constants(
            backend._reset_scratch_data, "reset_scratch", backend._warp.array
        ),
        "substep_views": [
            {
                "joint_names": list(view.joint_names),
                "body_names": list(view.body_names),
                "history_length": view.history_length,
            }
            for view in backend._physics_step_tensor_views
        ],
    }


def _metadata(array: Any, array_type: Any) -> dict[str, Any]:
    if isinstance(array, np.ndarray):
        return {"kind": "numpy", "shape": list(array.shape), "dtype": array.dtype.str}
    if not isinstance(array, array_type):
        raise TypeError("Continuation inventory contains a non-array")
    # Warp vector/matrix element shape is separate from its NumPy storage shape.
    return {"kind": "warp", "shape": list(array.shape), "dtype": str(array.dtype)}


def export_continuation_state(backend: Any) -> dict[str, Any]:
    """Make one detached CPU copy per field, with at most one temporary field copy."""
    _require_supported(backend)
    backend._synchronize()
    arrays = _arrays(backend)
    values = {
        name: value.copy() if isinstance(value, np.ndarray) else value.numpy().copy()
        for name, value in arrays.items()
    }
    return {
        "schema": SCHEMA,
        "identity": _identity(backend),
        "array_metadata": {
            name: _metadata(value, backend._warp.array) for name, value in arrays.items()
        },
        "arrays": values,
        "flags": {name: getattr(backend, name) for name in _FLAGS},
        "dr_mirror_stale": sorted(backend._dr_mirror_stale),
        "nbytes_by_field": {name: value.nbytes for name, value in values.items()},
        "nbytes": sum(value.nbytes for value in values.values()),
    }


def stage_array_payload(
    arrays: Mapping[str, Any],
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
    array_type: Any,
) -> list[tuple[Any, np.ndarray]]:
    """Validate numeric payload against constructor-owned allocations.

    No arrays are allocated from snapshot shapes. This shared core accepts real
    CPU native carriers for byte audits without claiming CUDA graph validation.
    Identity, synchronization and fault ownership remain with the caller.
    """
    if not isinstance(values, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("MJWarp continuation arrays and metadata must be mappings")
    if set(values) != set(arrays) or set(metadata) != set(arrays):
        raise ValueError("MJWarp continuation array inventory mismatch")
    for name, target in arrays.items():
        value = values[name]
        if metadata[name] != _metadata(target, array_type):
            raise ValueError(f"MJWarp continuation native layout mismatch: {name}")
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(f"MJWarp continuation requires numeric NumPy arrays: {name}")
        # A single native field is downloaded at a time, never a second complete
        # backend snapshot. No data writes happen during this validation phase.
        reference = target if isinstance(target, np.ndarray) else target.numpy()
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"MJWarp continuation storage layout mismatch: {name}")
        if isinstance(target, np.ndarray) and not target.flags.writeable:
            if target.tobytes() != value.tobytes():
                raise ValueError(f"MJWarp continuation immutable host array mismatch: {name}")
        del reference
    return [(target, values[name]) for name, target in arrays.items()]


def restore_staged_arrays(staged: list[tuple[Any, np.ndarray]]) -> None:
    """Apply a validated payload without replacing any constructor allocation."""
    for target, value in staged:
        if isinstance(target, np.ndarray):
            if target.flags.writeable:
                np.copyto(target, value)
        else:
            target.assign(value)


def import_continuation_state(backend: Any, state: Mapping[str, Any]) -> None:
    """Validate the entire payload before copying into existing native allocations."""
    _require_supported(backend)
    expected_keys = {
        "schema",
        "identity",
        "array_metadata",
        "arrays",
        "flags",
        "dr_mirror_stale",
        "nbytes_by_field",
        "nbytes",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys or state["schema"] != SCHEMA:
        raise ValueError("Invalid MJWarp continuation schema or fields")
    if state["identity"] != _identity(backend):
        raise ValueError("MJWarp continuation model, runtime, graph or substep identity mismatch")
    _import_continuation_payload(backend, state)


def _import_continuation_payload(backend: Any, state: Mapping[str, Any]) -> None:
    """Shared numeric core after the production identity and capability gates."""
    arrays = _arrays(backend)
    values, metadata = state["arrays"], state["array_metadata"]
    staged = stage_array_payload(arrays, values, metadata, backend._warp.array)
    flags = state["flags"]
    if not isinstance(flags, Mapping) or set(flags) != set(_FLAGS):
        raise ValueError("MJWarp continuation flag inventory mismatch")
    for name in _FLAGS:
        if type(flags[name]) is not type(getattr(backend, name)):
            raise ValueError(f"MJWarp continuation flag type mismatch: {name}")
    if flags["_completion_sequence"] < 0:
        raise ValueError("MJWarp continuation completion sequence must be non-negative")
    stale = state["dr_mirror_stale"]
    if not isinstance(stale, list) or any(not isinstance(name, str) for name in stale):
        raise ValueError("MJWarp continuation DR mirror names must be strings")
    if not set(stale).issubset({*EXPANDED_MODEL_FIELDS, *OPTION_VEC_FIELDS}):
        raise ValueError("MJWarp continuation contains unknown DR mirrors")
    sizes = {name: value.nbytes for name, value in values.items()}
    if state["nbytes_by_field"] != sizes or state["nbytes"] != sum(sizes.values()):
        raise ValueError("MJWarp continuation byte accounting mismatch")
    backend._synchronize()
    try:
        restore_staged_arrays(staged)
        backend._synchronize()
        for name in _FLAGS:
            setattr(backend, name, flags[name])
        backend._dr_mirror_stale = set(stale)
    except BaseException:
        backend._entity_faulted = True
        raise
