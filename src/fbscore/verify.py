"""``fbscore verify``: recompute a published settlement from its export and compare.

Exit status: 0 = settlement reproduced, 1 = mismatch (differences printed),
2 = the export could not be read or is malformed.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from . import __version__
from .canonicalize import CanonicalizationError, check_canonical
from .export import FORMAT, FORMAT_VERSION, load_export
from .grid import space_from_dict
from .rules import for_scoring_version
from .settlement import Settlement, Submission

__all__ = ["VerifyResult", "main", "verify_document"]

REL_TOL = 1e-9
ABS_TOL = 1e-9


class ExportError(ValueError):
    """The export is malformed (exit status 2)."""


@dataclass
class VerifyResult:
    mismatches: list[str] = field(default_factory=lambda: [])
    n_submissions: int = 0
    n_principals: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _get(d: Any, key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise ExportError(f"{where}: missing {key!r}")
    v: Any = cast("dict[str, Any]", d)[key]
    if isinstance(v, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,)):
        raise ExportError(f"{where}.{key}: wrong type")
    if not isinstance(v, kind):
        raise ExportError(f"{where}.{key}: wrong type")
    return v


def _obj(v: Any, where: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise ExportError(f"{where}: expected an object")
    return cast("dict[str, Any]", v)


def _objs(v: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(v, list):
        raise ExportError(f"{where}: expected an array")
    return [_obj(item, f"{where}[{i}]") for i, item in enumerate(cast("list[Any]", v))]


def _time(s: Any, where: str) -> datetime:
    if not isinstance(s, str):
        raise ExportError(f"{where}: expected an ISO-8601 timestamp")
    try:
        t = datetime.fromisoformat(s)
    except ValueError as e:
        raise ExportError(f"{where}: {e}") from e
    if t.tzinfo is None:
        raise ExportError(f"{where}: timestamp must include a UTC offset")
    return t


def _recompute(doc: dict[str, Any], result: VerifyResult) -> Settlement:
    if doc.get("format") != FORMAT or doc.get("format_version") != FORMAT_VERSION:
        raise ExportError(f"not a {FORMAT} v{FORMAT_VERSION} document")
    q = _get(doc, "question", dict, "$")
    try:
        space = space_from_dict(_get(q, "outcome_space", dict, "question"))
        prior = check_canonical(space, _get(q, "prior", list, "question"))
    except (ValueError, CanonicalizationError) as e:
        raise ExportError(f"question: {e}") from e
    scoring = _get(q, "scoring", dict, "question")
    try:
        rules = for_scoring_version(_get(scoring, "scoring_version", str, "question.scoring"))
    except ValueError as e:
        raise ExportError(str(e)) from e
    rule = _get(scoring, "rule", str, "question.scoring")
    if rule != rules.rule_name(space):
        result.mismatches.append(f"question.scoring.rule: {rule!r} but the rules use {rules.rule_name(space)!r}")
    k = float(_get(scoring, "k", (int, float), "question.scoring"))
    reserve = float(_get(scoring, "reserve", (int, float), "question.scoring"))
    decay = float(_get(scoring, "decay", (int, float), "question.scoring"))
    mode = _get(q, "payout_mode", str, "question")
    if mode not in ("cash", "credit_prize", "sandbox"):
        raise ExportError(f"question.payout_mode: unknown mode {mode!r}")
    pool_minor = _get(q, "prize_pool_minor", int, "question")
    open_at = _time(q.get("open_at"), "question.open_at")
    close_at = _time(q.get("close_at"), "question.close_at")
    cmax = rules.c_max(space, prior, k=k, reserve=reserve, prize_pool=pool_minor / 100)
    published_cmax = float(_get(q, "c_max", (int, float), "question"))
    if not _close(cmax, published_cmax):
        result.mismatches.append(f"question.c_max: published {published_cmax!r}, recomputed {cmax!r}")

    subs: list[Submission] = []
    for i, s in enumerate(_objs(_get(doc, "submissions", list, "$"), "submissions")):
        where = f"submissions[{i}]"
        try:
            pmf = check_canonical(space, _get(s, "pmf", list, where))
        except CanonicalizationError as e:
            raise ExportError(f"{where}.pmf: {e}") from e
        try:
            c = rules.c_at(
                c_max=cmax,
                decay=decay,
                open_at=open_at,
                close_at=close_at,
                at=_time(s.get("submitted_at"), f"{where}.submitted_at"),
            )
        except ValueError as e:
            raise ExportError(f"{where}: {e}") from e
        published_c = float(_get(s, "c", (int, float), where))
        if not _close(c, published_c):
            result.mismatches.append(f"{where}.c: published {published_c!r}, recomputed {c!r}")
        subs.append(
            Submission(
                _get(s, "seq", int, where),
                _get(s, "principal_id", str, where),
                _get(s, "forecaster_id", str, where),
                pmf,
                c,
            )
        )
    x_raw = _get(_get(doc, "outcome", dict, "$"), "x", (int, float), "outcome")
    x = float(x_raw)
    result.n_submissions = len(subs)
    result.n_principals = len({s.principal_id for s in subs})
    try:
        if mode == "cash":
            return rules.settle_cash(space, prior, subs, x, prize_pool_minor=pool_minor)
        if mode == "sandbox":
            return rules.settle_sandbox(space, prior, subs, x)
        # Credit prize: the remaining monthly mint cap is private, so it is not exported. Each
        # published mint is accepted as the cap; verify checks it never exceeds floor(net).
        lines = _objs(_get(_get(doc, "settlement", dict, "$"), "lines", list, "settlement"), "settlement.lines")
        caps = {str(ln.get("principal_id")): _get(ln, "credits_earned_minor", int, "settlement.lines") for ln in lines}
        for p in {s.principal_id for s in subs}:
            caps.setdefault(p, 0)
        return rules.settle_credit_prize(space, prior, subs, x, remaining_mint_cap_minor=caps)
    except ValueError as e:
        raise ExportError(str(e)) from e


def _compare(doc: dict[str, Any], st: Settlement, result: VerifyResult) -> None:
    pub = _get(doc, "settlement", dict, "$")
    if pub.get("payout_mode") != st.payout_mode:
        result.mismatches.append(
            f"settlement.payout_mode: published {pub.get('payout_mode')!r}, question says {st.payout_mode!r}"
        )
    for key, val in (("net_total", st.net_total), ("gross_positive", st.gross_positive), ("haircut", st.haircut)):
        got = float(_get(pub, key, (int, float), "settlement"))
        if not _close(got, val):
            result.mismatches.append(f"settlement.{key}: published {got!r}, recomputed {val!r}")
    for key, ival in (("prizes_total_minor", st.prizes_total_minor), ("refund_minor", st.refund_minor)):
        got_i = _get(pub, key, int, "settlement")
        if got_i != ival:
            result.mismatches.append(f"settlement.{key}: published {got_i}, recomputed {ival}")

    pub_scores: dict[Any, dict[str, Any]] = {
        s.get("seq"): s for s in _objs(_get(pub, "scores", list, "settlement"), "settlement.scores")
    }
    for row in st.scores:
        ps = pub_scores.pop(row.seq, None)
        if ps is None:
            result.mismatches.append(f"score for seq {row.seq}: missing from the published settlement")
            continue
        for key, val in (("points", row.points), ("norm_points", row.norm_points)):
            got = float(_get(ps, key, (int, float), f"scores[seq={row.seq}]"))
            if not _close(got, val):
                result.mismatches.append(f"score seq {row.seq}.{key}: published {got!r}, recomputed {val!r}")
    for seq in sorted(pub_scores, key=str):
        result.mismatches.append(f"score for seq {seq}: published but no such submission")

    pub_lines: dict[Any, dict[str, Any]] = {
        ln.get("principal_id"): ln for ln in _objs(_get(pub, "lines", list, "settlement"), "settlement.lines")
    }
    for line in st.lines:
        pl = pub_lines.pop(line.principal_id, None)
        where = f"line[{line.principal_id}]"
        if pl is None:
            result.mismatches.append(f"{where}: missing from the published settlement")
            continue
        got = float(_get(pl, "net_points", (int, float), where))
        if not _close(got, line.net_points):
            result.mismatches.append(f"{where}.net_points: published {got!r}, recomputed {line.net_points!r}")
        for key in ("prize_minor", "credits_burned_minor", "credits_earned_minor", "lock_minor"):
            want: int = getattr(line, key)
            got_i = _get(pl, key, int, where)
            if got_i != want:
                result.mismatches.append(f"{where}.{key}: published {got_i}, recomputed {want}")
    for p in sorted(pub_lines, key=str):
        result.mismatches.append(f"line[{p}]: published but the principal has no submissions")


def verify_document(doc: dict[str, Any]) -> VerifyResult:
    """Recompute and compare. Raises :class:`ExportError` if the document is malformed."""
    result = VerifyResult()
    st = _recompute(doc, result)
    _compare(doc, st, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fbscore", description="Tatnem scoring tools")
    parser.add_argument("--version", action="version", version=f"tatnem-scoring {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="recompute a settlement export and compare it with the published settlement")
    v.add_argument("export", help="settlement export (.json or .parquet)")
    args = parser.parse_args(argv)

    try:
        doc = load_export(args.export)
        if not isinstance(doc, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ExportError("top level must be an object")
        result = verify_document(doc)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if result.ok:
        print(f"OK: settlement reproduced ({result.n_submissions} submissions, {result.n_principals} principals)")
        return 0
    print(f"MISMATCH: {len(result.mismatches)} difference(s)")
    for m in result.mismatches:
        print(f"  - {m}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
