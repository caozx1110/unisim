from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from unisim import MotrixBackend, assert_backend_conformance
from unisim.dr import INTERVAL_TERM_PUSH, IntervalRandomizationPlan, IntervalTermOp
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-1 1'/></actuator>
</mujoco>"""


def test_motrix_backend_contract(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MotrixBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    assert_backend_conformance(backend)
    assert not backend.get_tensor_capabilities().packed_reset
    backend._host_step(np.zeros((2, 1)))
    state = backend._host_get_state()
    widths = backend.get_public_state_widths()
    assert state["qpos"].shape == (2, 7 + backend.num_dof_vel)
    assert state["qpos"].shape == (2, widths.nq)
    assert state["qvel"].shape == (2, widths.nv)


def test_non_portable_body_torque_fails_closed(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MotrixBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    try:
        zero = np.zeros((2, 1, 3), dtype=np.float32)
        with pytest.raises(NotImplementedError, match="interval body torque"):
            backend.apply_body_force(np.asarray([0]), zero, torque=zero)
    finally:
        backend.close()


def test_interval_push_uses_final_host_payload_without_global_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MotrixBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    try:
        force = np.asarray([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
        calls: list[tuple[np.ndarray, bool]] = []

        def add_external_force(data: object, applied_force: np.ndarray, local: bool) -> None:
            calls.append((np.array(applied_force, copy=True), local))

        class FakePushLink:
            def add_external_force(
                self, data: object, applied_force: np.ndarray, local: bool
            ) -> None:
                add_external_force(data, applied_force, local)

        monkeypatch.setattr(backend, "_push_body_link", FakePushLink())
        monkeypatch.setattr(np.random, "rand", pytest.fail, raising=True)
        backend.push_robots(force)
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(ops=(IntervalTermOp(INTERVAL_TERM_PUSH, force.copy()),))
        )

        assert [local for _, local in calls] == [True, True]
        for applied_force, _ in calls:
            np.testing.assert_array_equal(applied_force, force)

        legacy = np.asarray([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match=r"push force must have shape \(2, 3\)"):
            backend.push_robots(legacy)
        with pytest.raises(ValueError, match="push.*ndim 2"):
            backend.apply_interval_randomization(
                IntervalRandomizationPlan(ops=(IntervalTermOp(INTERVAL_TERM_PUSH, legacy),))
            )
        with pytest.raises(ValueError, match="NaN or Inf"):
            backend.push_robots(np.full((2, 3), np.nan))
    finally:
        backend.close()
