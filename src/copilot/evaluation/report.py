"""Markdown scorecard and regression gate."""

import datetime as dt
import json
from pathlib import Path

from copilot.evaluation.suites import LOWER_IS_BETTER, SuiteResult


def flatten(results: list[SuiteResult]) -> dict[str, float]:
    return {k: v for r in results for k, v in r.metrics.items()}


def gate(
    current: dict[str, float], baseline: dict[str, float], tolerance: float
) -> tuple[list[str], list[str]]:
    """(regressions, not-measured) comparing current metrics to a baseline.

    A metric regresses when it is worse than baseline by more than `tolerance` (absolute).
    Baseline metrics absent from this run (suite skipped) are reported, not failed.
    """
    regressions, missing = [], []
    for name, base in sorted(baseline.items()):
        if name not in current:
            missing.append(name)
            continue
        delta = current[name] - base
        worse = delta > tolerance if name in LOWER_IS_BETTER else delta < -tolerance
        if worse:
            regressions.append(f"{name}: {current[name]:.3f} vs baseline {base:.3f}")
    return regressions, missing


def scorecard(
    results: list[SuiteResult],
    skipped: dict[str, str],
    baseline: dict[str, float] | None = None,
    regressions: list[str] | None = None,
) -> str:
    out = [f"# Evaluation scorecard\n\n_Generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC_\n"]
    if regressions is not None:
        out.append(
            "**Gate: FAILED**\n\n" + "\n".join(f"- {r}" for r in regressions) + "\n"
            if regressions
            else "**Gate: passed** (no metric regressed beyond tolerance)\n"
        )
    for r in results:
        out.append(f"## {r.name}\n")
        out.append("| metric | value | baseline |\n|---|---:|---:|")
        for name, value in r.metrics.items():
            base = (baseline or {}).get(name)
            arrow = " (lower is better)" if name in LOWER_IS_BETTER else ""
            out.append(
                f"| `{name}`{arrow} | {value:.3f} | {'' if base is None else f'{base:.3f}'} |"
            )
        out.extend(f"\n_{n}_" for n in r.notes)
        failures = [row for row in r.rows if _row_failed(r.name, row)]
        if failures:
            out.append(f"\n<details><summary>{len(failures)} item(s) with problems</summary>\n")
            out.extend(f"- `{row.get('id')}`: {_row_summary(r.name, row)}" for row in failures)
            out.append("\n</details>")
        out.append("")
    for name, reason in skipped.items():
        out.append(f"## {name}\n\n_Skipped: {reason}_\n")
    return "\n".join(out)


def _row_failed(suite: str, row: dict) -> bool:
    if suite == "retrieval":
        return row.get("mode") == "hybrid+rerank" and row["hit"] == 0
    if suite == "generation":
        return (
            not row["correct_decision"]
            or row.get("key_fact_coverage", 1) < 1
            or bool(row.get("unsupported_entities"))
            or bool(row.get("unsupported_claims"))
        )
    if suite == "agent":
        return bool(
            row["missing"]
            or row["extra"]
            or row["wrong_severity"]
            or row["safety_failures"]
            or row["interaction_failures"]
        )
    return not row.get("validated", False)


def _row_summary(suite: str, row: dict) -> str:
    keys = {
        "retrieval": ["mode", "recall", "mrr"],
        "generation": [
            "expect",
            "refusal_reason",
            "key_fact_coverage",
            "unsupported_entities",
            "unsupported_claims",
        ],
        "agent": ["missing", "extra", "wrong_severity", "safety_failures", "interaction_failures"],
    }.get(suite, ["status", "error", "issues", "coverage"])
    return "; ".join(f"{k}={row[k]}" for k in keys if row.get(k) not in (None, [], ""))


def write(out_dir: Path, results: list[SuiteResult], markdown: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.md").write_text(markdown)
    (out_dir / "results.json").write_text(
        json.dumps(
            {"metrics": flatten(results), "rows": {r.name: r.rows for r in results}},
            indent=1,
            default=str,
        )
    )
    return out_dir / "scorecard.md"
