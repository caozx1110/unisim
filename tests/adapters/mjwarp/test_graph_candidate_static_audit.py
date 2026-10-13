"""Static audit for the MJWarp K=1 collector graph-candidate boundary."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[3] / "src" / "unisim" / "backend" / "mjwarp" / "backend.py"
)

AUDITED_METHODS = ("step_tensor", "set_state_tensor_mask")

FORBIDDEN_ALLOCATION_CALLS = frozenset({"empty", "zeros", "ones", "cat", "stack"})
FORBIDDEN_HOST_CALLS = frozenset({"cpu", "tolist", "numpy", "item", "nonzero"})


@dataclass(frozen=True)
class Violation:
    method: str
    line: int
    kind: str
    expression: str


def _methods(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    result: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MjwarpBackend":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in AUDITED_METHODS:
                    result[item.name] = item
    return result


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _static_violations(tree: ast.Module, synchronize_lines: set[int]) -> list[Violation]:
    violations: list[Violation] = []
    methods = _methods(tree)
    # Current stream captures are metadata-only in preflight, but the audited
    # public hot methods must not contain them.
    for name, method in methods.items():
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            called = _call_name(node)
            expression = _unparse(node)
            if called == "synchronize" and node.lineno not in synchronize_lines:
                violations.append(Violation(name, node.lineno, "host-sync", expression))
            if called == "current_stream":
                violations.append(Violation(name, node.lineno, "stream-query", expression))
            if called in FORBIDDEN_HOST_CALLS:
                violations.append(Violation(name, node.lineno, "host-detour", expression))
            if called in FORBIDDEN_ALLOCATION_CALLS and not any(
                keyword.arg == "out" for keyword in node.keywords
            ):
                violations.append(
                    Violation(name, node.lineno, "capture-time-allocation", expression)
                )
    return violations


# Completion is now represented by the public GraphCompletionToken metadata
# receipt. The hot methods must not synchronize a Torch stream at all.
DECLARED_COMPLETION_BARRIERS: frozenset[int] = frozenset()


def test_mjwarp_candidate_methods_are_present() -> None:
    methods = _methods(ast.parse(SOURCE.read_text(encoding="utf-8")))
    assert set(methods) == set(AUDITED_METHODS)


def test_mjwarp_candidate_static_audit_is_zero() -> None:
    violations = _static_violations(
        ast.parse(SOURCE.read_text(encoding="utf-8")), DECLARED_COMPLETION_BARRIERS
    )
    assert not violations, "MJWarp candidate static audit is nonzero:\n" + "\n".join(
        f"{item.method}:{item.line} {item.kind}: {item.expression}" for item in violations
    )
