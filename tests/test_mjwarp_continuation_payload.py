"""Numeric core preserves cold allocations and validates before any write."""

import numpy as np
import pytest

from unisim.backend.mjwarp.continuation import restore_staged_arrays, stage_array_payload


def test_cpu_cold_arrays_restore_raw_bytes_and_keep_addresses():
    targets = {
        "warmstart": np.zeros((3, 2), dtype=np.float32),
        "counter": np.zeros(3, dtype=np.int32),
    }
    values = {
        "warmstart": np.array([[np.nan, -0.0], [2, 3], [4, 5]], dtype=np.float32),
        "counter": np.array([19, 20, 21], dtype=np.int32),
    }
    metadata = {
        name: {"kind": "numpy", "shape": list(value.shape), "dtype": value.dtype.str}
        for name, value in targets.items()
    }
    addresses = {name: value.__array_interface__["data"][0] for name, value in targets.items()}
    staged = stage_array_payload(targets, values, metadata, ())
    restore_staged_arrays(staged)
    for name in targets:
        assert targets[name].tobytes() == values[name].tobytes()
        assert targets[name].__array_interface__["data"][0] == addresses[name]


def test_late_invalid_field_does_not_partially_write():
    targets = {"a": np.zeros(2, np.float32), "b": np.zeros(3, np.int32)}
    values = {"a": np.ones(2, np.float32), "b": np.ones(4, np.int32)}
    metadata = {
        name: {"kind": "numpy", "shape": list(value.shape), "dtype": value.dtype.str}
        for name, value in targets.items()
    }
    with pytest.raises(ValueError, match="storage layout mismatch: b"):
        stage_array_payload(targets, values, metadata, ())
    assert not targets["a"].any()


def test_immutable_array_must_already_match():
    target = np.zeros(2, np.int32)
    target.flags.writeable = False
    metadata = {"fixed": {"kind": "numpy", "shape": [2], "dtype": target.dtype.str}}
    with pytest.raises(ValueError, match="immutable host array"):
        stage_array_payload({"fixed": target}, {"fixed": np.ones(2, np.int32)}, metadata, ())
