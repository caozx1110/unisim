"""SDK-free checks for transactional validation and fixed-address restoration."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.base import SimBackend
from unisim.backend.mjwarp import continuation
from unisim.backend.mjwarp.randomization import EXPANDED_MODEL_FIELDS


class NativeArray:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float32)
        self.shape = self.values.shape
        self.dtype = "float32"
        self.fail = False

    def numpy(self):
        return self.values

    def assign(self, values):
        if self.fail:
            raise RuntimeError("native copy failed")
        np.copyto(self.values, values)


@dataclass
class Contact:
    force: NativeArray


@dataclass
class Data:
    qpos: NativeArray
    qacc_warmstart: NativeArray
    tree_asleep: NativeArray
    contact: Contact
    nworld: int = 2


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(continuation, "version", lambda package: "pinned-" + package)
    data = Data(
        *(NativeArray([index, index + 1]) for index in range(3)), Contact(NativeArray([4, 5]))
    )
    view = SimpleNamespace(
        joint_names=("joint",),
        body_names=("foot",),
        history_length=3,
        _before=NativeArray([6, 7]),
        _acc=NativeArray([8, 9]),
        _history=NativeArray([[10, 11, 12], [13, 14, 15]]),
    )
    result = SimpleNamespace(
        _warp=SimpleNamespace(array=NativeArray),
        _device_data=data,
        _device_model=SimpleNamespace(
            **{name: NativeArray([1, 2]) for name in EXPANDED_MODEL_FIELDS},
            opt=SimpleNamespace(gravity=NativeArray([0, 0, -9.81])),
            callback=SimpleNamespace(control=None),
        ),
        _reset_scratch_data=None,
        _physics_step_tensor_views=(view,),
        _sensor_cache=np.array([11, 12], dtype=np.float32),
        _xfrc_staging=np.array([13, 14], dtype=np.float32),
        _reset_mask_device=NativeArray([0, 1]),
        _dr_body_mass=np.array([1, 2], dtype=np.float32),
        _continuation_model_fingerprint="frozen-model",
        _num_envs=2,
        _sim_dt=0.002,
        _nconmax=32,
        _njmax=64,
        _cuda_graph_enabled=True,
        _tracked_body_state_dirty=False,
        _tracked_body_host_dirty=True,
        _host_cache_stale=True,
        _xfrc_pending=True,
        _device_xfrc_pending=False,
        _completion_sequence=17,
        _dr_mirror_stale={"body_mass"},
        _entity_faulted=False,
        _fixed_variant_plan=None,
        _pre_step_control_fn=None,
        _require_entity_healthy=lambda: None,
        _synchronize=lambda: None,
    )
    return result


def test_snapshot_is_detached_and_restore_preserves_all_array_addresses(backend):
    state = continuation.export_continuation_state(backend)
    arrays = continuation._arrays(backend)
    original_ids = {name: id(value) for name, value in arrays.items()}
    for value in arrays.values():
        (value if isinstance(value, np.ndarray) else value.values).fill(99)
    backend._completion_sequence = 30
    backend._dr_mirror_stale.clear()
    continuation.import_continuation_state(backend, state)
    assert original_ids == {
        name: id(value) for name, value in continuation._arrays(backend).items()
    }
    for name, value in arrays.items():
        actual = value if isinstance(value, np.ndarray) else value.numpy()
        np.testing.assert_array_equal(actual, state["arrays"][name])
    assert backend._completion_sequence == 17
    assert backend._dr_mirror_stale == {"body_mass"}
    assert state["nbytes"] == sum(value.nbytes for value in state["arrays"].values())
    assert "data.qacc_warmstart" in state["arrays"]
    assert "data.tree_asleep" in state["arrays"]
    assert "data.contact.force" in state["arrays"]
    assert "substep.0._history" in state["arrays"]


@pytest.mark.parametrize("problem", ["shape", "dtype", "missing", "identity", "flag", "bytes"])
def test_late_validation_error_does_not_partially_restore(backend, problem):
    state = continuation.export_continuation_state(backend)
    backend._device_data.qpos.values.fill(89)
    if problem == "shape":
        state["arrays"]["substep.0._history"] = np.ones(7, dtype=np.float32)
    elif problem == "dtype":
        state["arrays"]["substep.0._history"] = state["arrays"]["substep.0._history"].astype(
            np.float64
        )
    elif problem == "missing":
        del state["arrays"]["substep.0._history"]
    elif problem == "identity":
        state["identity"]["num_envs"] = 3
    elif problem == "flag":
        state["flags"]["_completion_sequence"] = -1
    else:
        state["nbytes"] += 1
    with pytest.raises(ValueError):
        continuation.import_continuation_state(backend, state)
    np.testing.assert_array_equal(backend._device_data.qpos.values, [89, 89])
    assert backend._entity_faulted is False


def test_partial_native_copy_failure_faults_backend(backend):
    state = continuation.export_continuation_state(backend)
    backend._device_data.contact.force.fail = True
    with pytest.raises(RuntimeError, match="native copy failed"):
        continuation.import_continuation_state(backend, state)
    assert backend._entity_faulted is True


def test_native_scalar_layout_change_is_rejected_before_write(backend):
    state = continuation.export_continuation_state(backend)
    backend._device_data.nworld = 3
    backend._device_data.qpos.values.fill(99)
    with pytest.raises(ValueError, match="identity mismatch"):
        continuation.import_continuation_state(backend, state)
    np.testing.assert_array_equal(backend._device_data.qpos.values, [99, 99])


def test_readonly_constructor_arrays_are_validated_without_write(backend):
    backend._immutable = np.broadcast_to(np.array([1], dtype=np.int64), (2,))
    state = continuation.export_continuation_state(backend)
    continuation.import_continuation_state(backend, state)
    state["arrays"]["adapter._immutable"][0] = 2
    with pytest.raises(ValueError, match="immutable host array"):
        continuation.import_continuation_state(backend, state)


def test_unknown_native_container_fails_closed(backend):
    backend._device_data.contact.force = object()
    with pytest.raises(TypeError, match="Unsupported native continuation field"):
        continuation.export_continuation_state(backend)


@pytest.mark.parametrize("unsupported", ["host_callback", "native_callback", "variant"])
def test_unserialized_external_state_is_rejected(backend, unsupported):
    state = continuation.export_continuation_state(backend)
    if unsupported == "host_callback":
        backend._pre_step_control_fn = lambda: None
    elif unsupported == "native_callback":
        backend._device_model.callback.control = lambda: None
    else:
        backend._fixed_variant_plan = object()
    with pytest.raises(NotImplementedError):
        continuation.export_continuation_state(backend)
    with pytest.raises(NotImplementedError):
        continuation.import_continuation_state(backend, state)


def test_unsupported_backend_does_not_fall_back_to_playback():
    owner = SimpleNamespace(backend_type="unsupported")
    with pytest.raises(NotImplementedError, match="continuation"):
        SimBackend.export_continuation_state(owner)
    with pytest.raises(NotImplementedError, match="continuation"):
        SimBackend.import_continuation_state(owner, {})
