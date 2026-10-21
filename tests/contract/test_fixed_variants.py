"""Backend-neutral fixed variant and reset payload metadata contract."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from unisim import FakeBackend
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
)
from unisim.scene import SceneCfg


def _plan(
    assignment: np.ndarray | None = None,
    *,
    layout: FixedVariantLayout = FixedVariantLayout.SAME_LAYOUT,
) -> FixedVariantPlan:
    return FixedVariantPlan(
        assignment=np.array([0, 1, 0], dtype=np.int32) if assignment is None else assignment,
        variants=(
            ModelSourceDescriptor("/cache/tools/tool-0.xml"),
            ModelSourceDescriptor("/cache/tools/tool-1.xml"),
        ),
        layout=layout,
    )


def test_fixed_variant_plan_is_validated_and_pickle_safe() -> None:
    original = _plan()
    assert original.layout is FixedVariantLayout.SAME_LAYOUT
    assert original.assignment.shape == (3,)
    assert not original.assignment.flags.writeable

    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert not restored.assignment.flags.writeable
    with np.testing.assert_raises(ValueError):
        restored.assignment[0] = 1


def _restore_from_state(cls, state):
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj


class _LegacyStatePickle:
    """Emit the historical state-dict pickle form: ``__new__`` plus ``__setstate__``."""

    def __init__(self, cls, state):
        self._cls = cls
        self._state = state

    def __reduce__(self):
        return (_restore_from_state, (self._cls, self._state))


def test_fixed_variant_plan_fresh_pickle_carries_only_public_assignment() -> None:
    original = _plan()
    payload = pickle.dumps(original)

    # Fresh pickles use the __reduce__ constructor form; the private reader-compat
    # key documented on __setstate__ must never be written by a current version.
    assert b"_assignment" not in payload
    cls, args = original.__reduce__()
    assert cls is FixedVariantPlan
    assignment, variants, layout = args
    np.testing.assert_array_equal(assignment, original.assignment)
    assert variants == original.variants
    assert layout is original.layout

    assert pickle.loads(payload) == original


def test_fixed_variant_plan_reads_historical_assignment_state_dicts() -> None:
    variants = (
        ModelSourceDescriptor("/cache/tools/tool-0.xml"),
        ModelSourceDescriptor("/cache/tools/tool-1.xml"),
    )
    for key in ("_assignment", "assignment"):
        legacy_array = np.array([1, 0, 1], dtype=np.int64)
        legacy_state = {
            key: legacy_array,
            "variants": variants,
            "layout": FixedVariantLayout.SAME_LAYOUT,
        }
        restored = pickle.loads(pickle.dumps(_LegacyStatePickle(FixedVariantPlan, legacy_state)))

        assert isinstance(restored, FixedVariantPlan)
        np.testing.assert_array_equal(restored.assignment, [1, 0, 1])
        assert not restored.assignment.flags.writeable
        assert restored.variants == variants
        assert restored.layout is FixedVariantLayout.SAME_LAYOUT

        # The legacy array storage is untrusted: restoring must copy it.
        legacy_array[0] = 0
        np.testing.assert_array_equal(restored.assignment, [1, 0, 1])


def test_fixed_variant_plan_rejects_mutable_and_out_of_range_inputs() -> None:
    with pytest.raises(TypeError, match="variants must be a tuple"):
        FixedVariantPlan(
            assignment=np.array([0]),
            variants=[ModelSourceDescriptor("tool.xml")],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="assignment must contain integers"):
        _plan(np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="values must be in"):
        _plan(np.array([0, 2, 0], dtype=np.int32))
    with pytest.raises(ValueError, match="non-empty"):
        _plan(np.array([], dtype=np.int32))
    with pytest.raises(ValueError, match="shape \\(2,\\)"):
        _plan().validate(2)


def test_model_source_descriptor_only_accepts_materialized_string_sources() -> None:
    with pytest.raises(TypeError, match="non-empty string"):
        ModelSourceDescriptor("")


def test_scene_cfg_carries_construction_time_fixed_variant_plan() -> None:
    plan = _plan()
    scene = SceneCfg(model_file="scene.xml", fixed_variant_plan=plan)

    assert scene.fixed_variant_plan is plan
    assert SceneCfg(model_file="scene.xml").fixed_variant_plan is None


def test_capabilities_negotiate_fixed_variant_layout() -> None:
    capabilities = DomainRandomizationCapabilities(
        supports_fixed_variants=True,
        supported_fixed_variant_layouts=frozenset({FixedVariantLayout.SAME_LAYOUT}),
    )
    assert capabilities.fixed_variant_rejections(_plan()) == ()

    uniform = _plan(layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT)
    assert capabilities.fixed_variant_rejections(uniform) == (
        "fixed variant layout 'uniform_public_layout' is unsupported",
    )

    unsupported = DomainRandomizationCapabilities()
    assert unsupported.fixed_variant_rejections(_plan()) == (
        "fixed variants are unsupported",
        "fixed variant layout 'same_layout' is unsupported",
    )


def test_reset_term_default_contract_fails_closed() -> None:
    backend = FakeBackend(num_envs=2, num_actuators=1)

    with pytest.raises(ValueError, match="unknown reset term"):
        backend.get_reset_term_default("not_a_reset_term")
    with pytest.raises(NotImplementedError, match="does not support reset term"):
        backend.get_reset_term_default("body_mass")
