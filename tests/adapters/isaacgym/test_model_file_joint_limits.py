"""MJCF joint-bound restoration for IsaacGym's model-file worker."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unisim.backend.isaacgym.scene_worker import SceneWorker
from unisim.backend.isaacgym.worker import _WorkerContext


def _props() -> dict[str, np.ndarray]:
    return {
        "driveMode": np.zeros(2, dtype=np.int32),
        "stiffness": np.zeros(2, dtype=np.float32),
        "damping": np.zeros(2, dtype=np.float32),
        "effort": np.zeros(2, dtype=np.float32),
        "armature": np.zeros(2, dtype=np.float32),
        "friction": np.zeros(2, dtype=np.float32),
        "hasLimits": np.zeros(2, dtype=bool),
        "lower": np.zeros(2, dtype=np.float32),
        "upper": np.zeros(2, dtype=np.float32),
    }


def _worker() -> _WorkerContext:
    worker = _WorkerContext.__new__(_WorkerContext)
    worker.gymapi = SimpleNamespace(DOF_MODE_POS=1)
    worker.num_dof = 2
    # Deliberately reverse the source order to cover importer reordering.
    worker.dof_names = ["second", "first"]
    return worker


def test_wire_payload_carries_source_joint_bounds() -> None:
    fields = _worker()._wire_dof_fields(
        {
            "dof_stiffness": [10.0, 20.0],
            "dof_damping": [1.0, 2.0],
            "dof_effort": [30.0, 40.0],
            "dof_armature": [0.1, 0.2],
            "dof_friction": [0.01, 0.02],
            "dof_lower": [-0.4, -0.8],
            "dof_upper": [0.8, 0.6],
        }
    )

    assert fields["lower"] == [-0.4, -0.8]
    assert fields["upper"] == [0.8, 0.6]


def test_actuator_properties_map_by_name_without_pretending_native_stops() -> None:
    worker = _worker()
    props = _props()
    fields = {
        "stiffness": [10.0, 20.0],
        "damping": [1.0, 2.0],
        "effort": [30.0, 40.0],
        "armature": [0.1, 0.2],
        "friction": [0.01, 0.02],
        "lower": [-0.4, -0.8],
        "upper": [0.8, 0.6],
    }

    worker._apply_actuator_props(props, fields, ["first", "second"])

    assert props["stiffness"].tolist() == [20.0, 10.0]
    assert props["damping"].tolist() == [2.0, 1.0]
    # Preview 4 ignores imported MJCF stops. The scene worker clamps targets
    # instead; this guard prevents a false native hard-limit support claim.
    assert not props["hasLimits"].any()


def test_control_targets_are_clamped_to_source_bounds() -> None:
    worker = SceneWorker.__new__(SceneWorker)
    worker.control_lower = np.array([[-1.0, -np.inf], [-0.6, -np.inf], [-0.4, -np.inf]])
    worker.control_upper = np.array([[0.4, np.inf], [0.6, np.inf], [0.8, np.inf]])
    values = np.array([[-1.2, 2.0]], dtype=np.float32)

    np.testing.assert_allclose(
        worker._clip_control_values(values, np.array([2])), [[-0.4, 2.0]], rtol=0, atol=0
    )


def test_partial_reset_control_targets_use_reset_environment_bounds() -> None:
    worker = SceneWorker.__new__(SceneWorker)
    worker.control_lower = np.array([[-1.0], [-0.6], [-0.4]])
    worker.control_upper = np.array([[0.4], [0.6], [0.8]])

    np.testing.assert_allclose(
        worker._clip_control_values(np.array([[-2.0], [2.0]], dtype=np.float32), np.array([0, 2])),
        [[-1.0], [0.8]],
        rtol=0,
        atol=0,
    )
