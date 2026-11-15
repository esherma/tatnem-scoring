"""Exit criterion: no abuse-relevant parameter has a library default (TDD §2.3.1, §12)."""

from __future__ import annotations

import ast
from pathlib import Path

from fbscore import ABUSE_PARAMETER_NAMES

SRC = Path(__file__).resolve().parents[1] / "src" / "fbscore"


def _functions() -> list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]]:
    out: list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        out.extend((path, n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef))
    return out


def _defaulted(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = fn.args
    positional = [*a.posonlyargs, *a.args]
    names = {p.arg for p in positional[len(positional) - len(a.defaults) :]}
    names |= {p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None}
    return names


def test_no_abuse_parameter_has_a_default() -> None:
    offenders = [
        f"{path.relative_to(SRC)}:{fn.lineno} {fn.name}({name}=...)"
        for path, fn in _functions()
        for name in _defaulted(fn) & ABUSE_PARAMETER_NAMES
    ]
    assert not offenders, "abuse parameters must be required arguments: " + ", ".join(offenders)


def test_every_listed_name_is_a_real_parameter() -> None:
    used = {p.arg for _, fn in _functions() for p in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
    assert used >= ABUSE_PARAMETER_NAMES, f"stale names: {sorted(ABUSE_PARAMETER_NAMES - used)}"


def test_no_module_level_constant_holds_an_abuse_value() -> None:
    upper = {n.upper() for n in ABUSE_PARAMETER_NAMES}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            for t in targets:
                if isinstance(t, ast.Name):
                    assert t.id.upper() not in upper, (
                        f"{path.name}: module constant {t.id} looks like an abuse parameter"
                    )
