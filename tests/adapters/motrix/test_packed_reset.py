"""Backend-owned packed reset transactions for the MotrixSim host bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

import torch  # noqa: E402

from unisim import MotrixBackend  # noqa: E402
from unisim.dr.types import (  # noqa: E402
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    TensorResetRandomizationPayload,
)
from unisim.scene import SceneCfg  # noqa: E402

MODEL = """<mujoco model='motrix-packed-reset'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='base' pos='0 0 1'>
      <freejoint name='root'/>
      <inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>
      <geom name='visual_geom' type='sphere' size='.10' contype='0' conaffinity='0'
            friction='.70 .006 .0001'/>
      <geom name='base_geom' type='sphere' size='.08'/>
      <body name='link' pos='0 0 .2'>
        <joint name='drive' type='hinge' axis='0 1 0'/>
        <inertial pos='0 0 0' mass='.2' diaginertia='.02 .02 .02'/>
        <geom name='link_geom' type='sphere' size='.03'/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name='drive' joint='drive'/></actuator>
</mujoco>"""


@pytest.fixture
def backend(tmp_path: Path) -> MotrixBackend:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    result = MotrixBackend(SceneCfg(str(model_path)), num_envs=3, sim_dt=0.002)
    try:
        yield result
    finally:
        result.close()


def _install_default_state(
    backend: MotrixBackend, transaction: object
) -> tuple[torch.Tensor, torch.Tensor]:
    state = backend._host_get_state()
    default_qpos = torch.tensor(state["qpos"][0], dtype=torch.float32)
    default_qvel = torch.tensor(state["qvel"][0], dtype=torch.float32)
    transaction.set_state_default(default_qpos, default_qvel)
    return default_qpos, default_qvel


def test_packed_reset_commits_model_fields_and_promotes_baseline(
    backend: MotrixBackend,
) -> None:
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.reset_randomization
    assert capabilities.device_reset_randomization
    assert capabilities.packed_reset
    assert RESET_TERM_BODY_MASS in capabilities.packed_reset_fields
    assert capabilities.mask_selected_reset
    assert not capabilities.packed_reset_mocap

    transaction = backend.bind_packed_reset(device="cpu")
    default_qpos, default_qvel = _install_default_state(backend, transaction)
    baseline = transaction.baseline(RESET_TERM_BODY_MASS).clone()
    body_id = int(backend.get_body_ids(("base",))[0])
    requested = baseline.clone()
    requested[1, body_id] = 2.5
    rows = torch.tensor([1], dtype=torch.int64)
    columns = torch.tensor([body_id], dtype=torch.int64)
    requested_values = requested.index_select(0, rows)[:, columns]
    transaction.write_model_field(
        RESET_TERM_BODY_MASS,
        rows,
        requested_values,
        columns=columns,
    )
    transaction.begin_state(rows)
    result = transaction.commit_state(
        rows,
        default_qpos.expand(1, -1),
        default_qvel.expand(1, -1),
    )
    assert result is not None
    assert result["timing"]["packed_reset_total_ms"] >= 0.0

    applied_mass = np.asarray(backend._body_link.get_mass_override(backend._data)).reshape(-1)
    np.testing.assert_allclose(applied_mass[1], 2.5, rtol=0.0)
    np.testing.assert_allclose(
        applied_mass[[0, 2]],
        baseline[[0, 2], body_id],
        rtol=0.0,
    )
    torch.testing.assert_close(
        transaction.baseline(RESET_TERM_BODY_MASS), requested, check_dtype=False
    )


def test_packed_reset_mask_commit_preserves_false_rows(backend: MotrixBackend) -> None:
    transaction = backend.bind_packed_reset(device="cpu")
    default_qpos, default_qvel = _install_default_state(backend, transaction)
    baseline = transaction.baseline(RESET_TERM_BODY_MASS)
    body_id = int(backend.get_body_ids(("base",))[0])
    requested = baseline.clone()
    requested[1, body_id] = 3.25
    reset_mask = torch.tensor([False, True, False])
    transaction.write_model_field_mask(RESET_TERM_BODY_MASS, reset_mask, requested)
    transaction.begin_state_mask(reset_mask)
    result = transaction.commit_mask(reset_mask)
    assert result is not None

    applied_mass = np.asarray(backend._body_link.get_mass_override(backend._data)).reshape(-1)
    np.testing.assert_allclose(applied_mass[1], 3.25, rtol=0.0)
    np.testing.assert_allclose(
        applied_mass[[0, 2]],
        baseline[[0, 2], body_id],
        rtol=0.0,
    )
    state = backend.get_state_views(device="cpu")
    torch.testing.assert_close(state["qpos"][1], default_qpos)
    torch.testing.assert_close(state["qvel"][1], default_qvel)


def test_packed_reset_geom_friction_preserves_visual_rows(backend: MotrixBackend) -> None:
    capabilities = backend.get_tensor_capabilities()
    assert RESET_TERM_GEOM_FRICTION in capabilities.packed_reset_fields

    visual_id = backend.get_geom_id("visual_geom")
    collision_id = backend.get_geom_id("base_geom")
    transaction = backend.bind_packed_reset(device="cpu")
    default_qpos, default_qvel = _install_default_state(backend, transaction)
    baseline = transaction.baseline(RESET_TERM_GEOM_FRICTION)
    np.testing.assert_allclose(
        backend.get_geom_friction()[visual_id],
        baseline[0, visual_id].numpy(),
        rtol=0.0,
        atol=1e-7,
    )
    requested = baseline.clone()
    requested[1, collision_id] = torch.tensor([0.9, 0.02, 0.0])
    rows = torch.tensor([1], dtype=torch.int64)
    columns = torch.tensor([collision_id], dtype=torch.int64)
    transaction.write_model_field(
        RESET_TERM_GEOM_FRICTION,
        rows,
        requested.index_select(0, rows)[:, columns],
        columns=columns,
    )
    transaction.begin_state(rows)
    result = transaction.commit_state(
        rows,
        default_qpos.expand(1, -1),
        default_qvel.expand(1, -1),
    )
    assert result is not None

    committed = backend.get_reset_term_default(RESET_TERM_GEOM_FRICTION)
    np.testing.assert_allclose(
        committed[1, collision_id], [0.9, 0.02, 0.0], rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        committed[:, visual_id], baseline[:, visual_id].numpy(), rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        committed[[0, 2], collision_id],
        baseline[[0, 2], collision_id].numpy(),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        transaction.baseline(RESET_TERM_GEOM_FRICTION), requested, check_dtype=False
    )


def test_packed_reset_visual_geom_friction_change_fails_closed(backend: MotrixBackend) -> None:
    visual_id = backend.get_geom_id("visual_geom")
    transaction = backend.bind_packed_reset(device="cpu")
    default_qpos, default_qvel = _install_default_state(backend, transaction)
    baseline = transaction.baseline(RESET_TERM_GEOM_FRICTION)
    requested = baseline.clone()
    requested[2, visual_id] += 0.1
    rows = torch.tensor([2], dtype=torch.int64)
    columns = torch.tensor([visual_id], dtype=torch.int64)
    transaction.write_model_field(
        RESET_TERM_GEOM_FRICTION,
        rows,
        requested.index_select(0, rows)[:, columns],
        columns=columns,
    )
    transaction.begin_state(rows)
    with pytest.raises(ValueError, match="only supports collision geoms"):
        transaction.commit_state(
            rows,
            default_qpos.expand(1, -1),
            default_qvel.expand(1, -1),
        )


@pytest.mark.parametrize("use_mask", [False, True])
def test_direct_tensor_geom_friction_reset_preserves_visual_rows(
    backend: MotrixBackend, use_mask: bool
) -> None:
    visual_id = backend.get_geom_id("visual_geom")
    collision_id = backend.get_geom_id("base_geom")
    default = backend.get_reset_term_default(RESET_TERM_GEOM_FRICTION)
    payload_values = torch.as_tensor(np.array(default), dtype=torch.float32).clone()
    payload_values[1, collision_id] = torch.tensor([0.85, 0.015, 0.0])
    payload = TensorResetRandomizationPayload(geom_friction=payload_values)

    state = backend.get_state_views(device="cpu")
    qpos = state["qpos"]
    qvel = state["qvel"]
    if use_mask:
        reset_mask = torch.tensor([False, True, False])
        backend.set_state_tensor_mask(reset_mask, qpos.contiguous(), qvel.contiguous(), payload)
    else:
        rows = torch.tensor([1], dtype=torch.int64)
        selected_payload = TensorResetRandomizationPayload(geom_friction=payload_values[[1]])
        backend.set_state_tensor(
            rows, qpos[[1]].contiguous(), qvel[[1]].contiguous(), selected_payload
        )

    committed = backend.get_reset_term_default(RESET_TERM_GEOM_FRICTION)
    np.testing.assert_allclose(
        committed[1, collision_id], [0.85, 0.015, 0.0], rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(committed[:, visual_id], default[:, visual_id], rtol=0.0, atol=1e-7)
