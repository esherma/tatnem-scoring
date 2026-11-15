"""No private platform material is published here (TDD §2.3.1).

The denylist is stored as SHA-256 hashes of tokens so that this public file does not
itself reveal the private names and values it guards against. Every word and number token
in the repository's text files is hashed and checked against it.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".hypothesis", ".ruff_cache", "dist", "fixtures"}
SKIP_FILES = {"uv.lock", "LICENSE"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".cfg", ".ini", ""}
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*[A-Za-z0-9_]|\d+(?:\.\d+)?%?")

DENYLIST_SHA256 = frozenset(
    {
        "a304be55708dcc9f071d0de56dc3a16541a62de869715de575aa1769e2d37da2",
        "45caabef79a5df01c2e83d9feb6acb1fa83efaec2b66c28bff0c7d724dbc79ee",
        "e04485a5d1c4a9f98cd604cf837519084e93766941fb0530b962bf59103dcc91",
        "4a137d360a9818873148014d60f04e6b5403e1c8f4adfc995e6f4e358a9a80ac",
        "02c545db43902bab8cbc906ecc9f605bc6db98cea2b3ab9044c0bbddec982412",
        "72da55d317fd997b93138b8646a2238e806a4c4566d1e854277b5a583d8aef23",
        "36819111cffb9e1e90eb1fb7c0c29b56a1bc148d41478af66a3de0e5486bb688",
        "1e472b39b105d349bcd069c4a711b44a2fffb8e274714bb07ecfff69a9a7f67b",
        "23ec5e7f20a2bcc3fca42ac6f74643066b90a0458cf4f309bc2dc517869c7f39",
        "6ea3880b29a0a586a59c9f550391ae532ca5f5a1bd43c6f2f93e2158ba672070",
        "60c716e2a10c9879776a6f2862aa531c1fe2fde7e73f001e25b533759a6fcdfd",
        "15d8c11e32e279f9116f82dfd583da9df1fd1397d2f55b7cce7bf143887e6cc7",
        "9bb48203991c2d2f1cff19ea84815ca9a37a92cac2f6c3d47a690bb3a3ec4230",
        "794c3a79ad64ee459ea8832671b4610c17cf8795b5f9e703350e2e4a9f61e250",
        "05ab886f80f031b80e35a07d76b9d51b079dcac39600d512e7aac3ffe6e3cd3b",
        "7b660c0e5c82ce621855d656178c608d0c0d625f8fc26b7d9674d529f0a345f4",
        "ab7ff122cd16b1e650a02bfe3654e9ada2a6e17a240abd93f62a7f13c92fba3e",
    }
)


def _candidates() -> list[Path]:
    """Files git would publish (tracked or not ignored); falls back to a directory walk."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        return [ROOT / n for n in out.decode().split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        return list(ROOT.rglob("*"))


def _files() -> list[Path]:
    out: list[Path] = []
    for p in _candidates():
        rel = p.relative_to(ROOT)
        if not p.is_file() or set(rel.parts) & SKIP_DIRS or p.name in SKIP_FILES or p.suffix not in TEXT_SUFFIXES:
            continue
        out.append(p)
    return out


def test_scan_covers_the_repository() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in _files()}
    assert {"spec/scoring.md", "src/fbscore/aggregate.py", "README.md"} <= names


def test_no_denylisted_token_is_published() -> None:
    hits: list[str] = []
    for path in _files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for tok in TOKEN.findall(line):
                for t in {tok, tok.lower(), tok.rstrip(".")}:
                    if hashlib.sha256(t.encode()).hexdigest() in DENYLIST_SHA256:
                        hits.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not hits, f"private material found at: {sorted(set(hits))}"
