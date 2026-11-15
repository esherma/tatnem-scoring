"""Settlement export format (``tatnem-settlement-export``, format version 0).

A settlement export carries every input needed to recompute a question's settlement:
frozen scoring parameters, grid, prior, ordered submissions with pmfs, times and c, the
outcome, and the published settlement. JSON is the canonical encoding
(``schemas/settlement-export.v0.schema.json``). Parquet is supported through the
``verify`` extra: one row per submission, everything else as JSON in the file's
key-value metadata under ``tatnem.export``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from .settlement import Settlement

__all__ = ["FORMAT", "FORMAT_VERSION", "build_export", "load_export", "write_parquet"]

FORMAT: Final = "tatnem-settlement-export"
FORMAT_VERSION: Final = "0"
PARQUET_METADATA_KEY: Final = b"tatnem.export"


def _iso(t: datetime) -> str:
    if t.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return t.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_export(
    *,
    question: Mapping[str, Any],
    submissions: list[Mapping[str, Any]],
    settlement: Settlement,
) -> dict[str, Any]:
    """Assemble an export document.

    ``question`` holds ``id``, ``outcome_space``, ``prior``, ``scoring`` (``rule``, ``k``,
    ``reserve``, ``decay``, ``scoring_version``), ``c_max``, ``open_at``, ``close_at``,
    ``payout_mode`` and ``prize_pool_minor``. Each submission holds ``seq``,
    ``forecaster_id``, ``principal_id``, ``submitted_at``, ``pmf`` and ``c``.
    """
    q = dict(question)
    for key in ("open_at", "close_at"):
        if isinstance(q[key], datetime):
            q[key] = _iso(q[key])
    subs: list[dict[str, Any]] = []
    for s in submissions:
        d = dict(s)
        if isinstance(d["submitted_at"], datetime):
            d["submitted_at"] = _iso(d["submitted_at"])
        d["pmf"] = list(d["pmf"])
        subs.append(d)
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "question": q,
        "outcome": {"x": settlement.outcome},
        "submissions": subs,
        "settlement": {
            "payout_mode": settlement.payout_mode,
            "net_total": settlement.net_total,
            "gross_positive": settlement.gross_positive,
            "haircut": settlement.haircut,
            "prizes_total_minor": settlement.prizes_total_minor,
            "refund_minor": settlement.refund_minor,
            "scores": [{"seq": r.seq, "points": r.points, "norm_points": r.norm_points} for r in settlement.scores],
            "lines": [
                {
                    "principal_id": ln.principal_id,
                    "net_points": ln.net_points,
                    "prize_minor": ln.prize_minor,
                    "credits_burned_minor": ln.credits_burned_minor,
                    "credits_earned_minor": ln.credits_earned_minor,
                    "lock_minor": ln.lock_minor,
                }
                for ln in settlement.lines
            ],
        },
    }


def write_parquet(doc: Mapping[str, Any], path: str | Path) -> None:
    """Write an export document as Parquet (requires ``pyarrow``)."""
    pa: Any = importlib.import_module("pyarrow")
    pq: Any = importlib.import_module("pyarrow.parquet")

    subs = doc["submissions"]
    scores = {s["seq"]: s for s in doc["settlement"]["scores"]}
    table: Any = pa.table(
        {
            "seq": pa.array([s["seq"] for s in subs], pa.int64()),
            "forecaster_id": pa.array([s["forecaster_id"] for s in subs], pa.string()),
            "principal_id": pa.array([s["principal_id"] for s in subs], pa.string()),
            "submitted_at": pa.array([s["submitted_at"] for s in subs], pa.string()),
            "pmf": pa.array([s["pmf"] for s in subs], pa.list_(pa.float64())),
            "c": pa.array([s["c"] for s in subs], pa.float64()),
            "points": pa.array([scores[s["seq"]]["points"] for s in subs], pa.float64()),
            "norm_points": pa.array([scores[s["seq"]]["norm_points"] for s in subs], pa.float64()),
        }
    )
    rest = {k: v for k, v in doc.items() if k != "submissions"}
    rest["settlement"] = {k: v for k, v in doc["settlement"].items() if k != "scores"}
    table = table.replace_schema_metadata({PARQUET_METADATA_KEY: json.dumps(rest, sort_keys=True).encode()})
    pq.write_table(table, str(path))


def _load_parquet(path: Path) -> dict[str, Any]:
    try:
        pq: Any = importlib.import_module("pyarrow.parquet")
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("reading Parquet needs the verify extra: pip install 'tatnem-scoring[verify]'") from e
    table: Any = pq.read_table(str(path))
    meta: dict[bytes, bytes] = table.schema.metadata or {}
    if PARQUET_METADATA_KEY not in meta:
        raise ValueError("Parquet file has no tatnem.export metadata")
    doc: dict[str, Any] = json.loads(meta[PARQUET_METADATA_KEY])
    rows: list[dict[str, Any]] = table.to_pylist()
    doc["submissions"] = [
        {k: r[k] for k in ("seq", "forecaster_id", "principal_id", "submitted_at", "pmf", "c")} for r in rows
    ]
    doc["settlement"]["scores"] = [
        {"seq": r["seq"], "points": r["points"], "norm_points": r["norm_points"]} for r in rows
    ]
    return doc


def load_export(path: str | Path) -> dict[str, Any]:
    """Load a JSON (``.json``) or Parquet (``.parquet``) export into a document dict."""
    p = Path(path)
    if p.suffix == ".parquet":
        return _load_parquet(p)
    with p.open(encoding="utf-8") as f:
        doc: dict[str, Any] = json.load(f)
    return doc
