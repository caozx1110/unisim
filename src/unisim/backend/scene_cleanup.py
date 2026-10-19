"""Shared scene-materialization helpers."""

from __future__ import annotations

import os


class TemporarySceneCleanup:
    """Own the temporary XMLs created while materializing one scene."""

    def __init__(self, *paths: str) -> None:
        self._paths = paths
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        for path in self._paths:
            remove_temporary_file(path)


def remove_temporary_file(
    path: str | os.PathLike[str], *, original: str | os.PathLike[str] | None = None
) -> None:
    """Remove one temporary scene artifact, ignoring an absent/unchanged file."""
    if original is not None and os.path.exists(path) and os.path.exists(original):
        if os.path.samefile(path, original):
            return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
