"""SDK-free checks for the external IsaacGym worker import boundary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from unisim.backend.isaacgym.dependencies import IsaacGymRuntime, build_worker_env
from unisim.backend.subprocess_ipc.worker_profile import (
    PROFILE_START_COMMAND_ENV,
    PROFILE_STOP_COMMAND_ENV,
    PROFILE_TRACE_ENV,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER_PATH = _REPO_ROOT / "src" / "unisim" / "backend" / "isaacgym" / "worker.py"
_PROTOCOL_PATH = _REPO_ROOT / "src" / "unisim" / "backend" / "subprocess_ipc" / "protocol.py"
_PROFILE_ENVIRONMENT = (PROFILE_TRACE_ENV, PROFILE_START_COMMAND_ENV, PROFILE_STOP_COMMAND_ENV)


def _import_worker_without_host_unisim() -> ModuleType:
    """Load the worker exactly as its Python 3.8 interpreter does: by file path."""

    class NoHostUniSim:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
            if fullname == "unisim" or fullname.startswith("unisim."):
                raise ModuleNotFoundError(f"host package unavailable to worker: {fullname}")
            return None

    spec = importlib.util.spec_from_file_location("unisim_isaacgym_worker_test", _WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    no_host_unisim = NoHostUniSim()
    sys.meta_path.insert(0, no_host_unisim)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.meta_path.remove(no_host_unisim)
    return module


@pytest.fixture(autouse=True)
def _no_worker_profiler_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROFILE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def test_worker_owner_does_not_import_host_unisim() -> None:
    assert _import_worker_without_host_unisim()


def test_worker_loads_protocol_by_file_path_without_site_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _import_worker_without_host_unisim()
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    protocol = worker._load_protocol(str(_PROTOCOL_PATH))
    assert protocol.__name__ == "unisim_subprocess_protocol"


def test_worker_profiler_still_loads_by_file_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _import_worker_without_host_unisim()
    monkeypatch.setenv(_PROFILE_ENVIRONMENT[0], "/tmp/isaacgym-worker-trace.json")
    monkeypatch.setenv(_PROFILE_ENVIRONMENT[1], "echo start")
    profiler = worker._load_worker_profiler(str(_PROTOCOL_PATH))
    assert profiler is not None


def test_worker_environment_does_not_inherit_host_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    runtime = IsaacGymRuntime(
        python=Path("/worker/bin/python3.8"),
        isaacgym_python=Path("/worker/isaacgym/python"),
        lib_path=Path("/worker/lib"),
    )

    environment = build_worker_env(runtime)

    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["LD_LIBRARY_PATH"].startswith("/worker/lib")
    assert environment["PATH"].startswith("/worker/bin")
