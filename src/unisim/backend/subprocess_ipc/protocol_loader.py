"""File-path bootstrap helpers for external subprocess workers.

The loaders are intentionally independent from :mod:`unisim` so isolated
Python 3.8 IsaacGym workers can copy or load this file without site-packages.
They never import an optional SDK and never register protocol modules in
``sys.modules``; each caller remains responsible for its dedicated runtime.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any


def protocol_module_from_path(path: str) -> Any:
    """Load the canonical protocol module by file path, not package identity."""
    spec = importlib.util.spec_from_file_location("unisim_subprocess_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_profiler_from_path(protocol_path: str) -> Any:
    """Load the opt-in worker profiler beside the dynamically loaded protocol."""
    profile_environment = (
        "UNISIM_ISAAC_WORKER_PROFILE_TRACE",
        "UNISIM_ISAAC_WORKER_PROFILE_START_COMMAND",
        "UNISIM_ISAAC_WORKER_PROFILE_STOP_COMMAND",
    )
    if not any(name in os.environ for name in profile_environment):
        return None
    path = os.path.join(os.path.dirname(protocol_path), "worker_profile.py")
    spec = importlib.util.spec_from_file_location("unisim_worker_profile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load worker profiler module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.profiler_from_environment()


__all__ = ["protocol_module_from_path", "worker_profiler_from_path"]
