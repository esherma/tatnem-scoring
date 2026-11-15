"""The README's Python example runs as written."""

from __future__ import annotations

import re
from pathlib import Path


def test_readme_example_runs() -> None:
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    assert match is not None
    exec(compile(match.group(1), "README.md", "exec"), {})  # noqa: S102
