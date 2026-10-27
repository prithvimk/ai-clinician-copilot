"""Run evaluation suites and write a markdown scorecard; optionally gate on regressions.

    uv run python eval/run_eval.py                         # agent rules only (no models/keys)
    uv run python eval/run_eval.py --suites retrieval,generation,agent --judge
    uv run python eval/run_eval.py --gate eval/baseline.json      # exit 1 on regression
    uv run python eval/run_eval.py --write-baseline eval/baseline.json
    uv run python eval/run_eval.py --propose "When to stop metformin?"   # annotate gold

Suites:
    agent       deterministic findings / safety / interaction checks (always runnable)
    retrieval   Recall@K, Precision@K, MRR, hit rate; ablation dense vs sparse vs hybrid vs
                hybrid+rerank (needs the index and embedding models)
    generation  refusal decisions, key-fact coverage, citation validity, entity cross-check;
                with --judge also faithfulness, citation accuracy, groundedness,
                hallucination rate and answer relevance (needs ANTHROPIC_API_KEY)
    agent_llm   full patient workflow with the real LLM (needs ANTHROPIC_API_KEY)
"""

import argparse
import json
import os
import sys
from pathlib import Path

from copilot.config import PROJECT_ROOT, get_settings
from copilot.evaluation.golden import load_agent, load_qa
from copilot.evaluation.report import flatten, gate, scorecard, write
from copilot.evaluation.suites import (
    run_agent_rules,
    run_agent_workflow,
    run_generation,
    run_retrieval,
)

ALL_SUITES = ("agent", "retrieval", "generation", "agent_llm")


def _need_key() -> str | None:
    if (
        get_settings().anthropic_api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        return None
    return "no Anthropic credentials (set ANTHROPIC_API_KEY)"


class RetrievalStack:
    """Loads the vector store and retriever once; remembers why it couldn't."""

    def __init__(self):
        self.error: str | None = None
        try:
            from copilot.rag.retriever import get_retriever
            from copilot.rag.vector_store import get_vector_store

            self.store, self.retriever = get_vector_store(), get_retriever()
            self.indexed = set(self.store.stats())
        except Exception as e:  # models unavailable, index locked, etc.
            self.error = f"retrieval stack unavailable: {type(e).__name__}: {e}"
            self.indexed = set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suites", default="agent", help=f"comma list of {', '.join(ALL_SUITES)}")
    parser.add_argument("--k", type=int, default=5, help="cutoff for retrieval metrics")
    parser.add_argument("--judge", action="store_true", help="run LLM-judge generation metrics")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "reports")
    parser.add_argument("--gate", type=Path, help="baseline JSON; exit 1 if any metric regresses")
    parser.add_argument("--tolerance", type=float, default=0.02, help="allowed absolute drop")
    parser.add_argument("--write-baseline", type=Path, help="save this run's metrics as baseline")
    parser.add_argument("--propose", metavar="QUESTION", help="print candidate gold chunks")
    args = parser.parse_args(argv)

    if args.propose:
        from copilot.rag.retriever import get_retriever

        for hit in get_retriever().retrieve(args.propose, k=args.k):
            c = hit.chunk
            print(f"{hit.score:.3f}  {c.doc_id}  {c.citation()}\n      {c.text[:200]!r}\n")
        return 0

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = set(suites) - set(ALL_SUITES)
    if unknown:
        parser.error(f"unknown suites: {', '.join(sorted(unknown))}")

    results, skipped = [], {}
    qa_items = load_qa()
    scenarios = load_agent()

    if "agent" in suites:
        results.append(run_agent_rules(scenarios))

    stack = RetrievalStack() if {"retrieval", "generation", "agent_llm"} & set(suites) else None

    if "retrieval" in suites:
        if stack.error:
            skipped["retrieval"] = stack.error
        elif not stack.indexed:
            skipped["retrieval"] = "index is empty (run the ingest CLI first)"
        else:
            searches = {
                mode: (lambda q, k, m=mode: stack.store.search(q, k=k, mode=m))
                for mode in ("dense", "sparse", "hybrid")
            }
            searches["hybrid+rerank"] = lambda q, k: stack.retriever.retrieve(q, k=k)
            results.append(run_retrieval(qa_items, searches, args.k, stack.indexed))

    for suite in ("generation", "agent_llm"):
        if suite not in suites:
            continue
        if reason := stack.error or _need_key():
            skipped[suite] = reason
            continue
        from copilot.rag.llm import get_llm

        if suite == "generation":
            from copilot.rag.qa import get_qa

            results.append(run_generation(qa_items, get_qa(), get_llm() if args.judge else None))
        else:
            results.append(run_agent_workflow(scenarios, stack.retriever, get_llm()))

    metrics = flatten(results)
    baseline = json.loads(args.gate.read_text()) if args.gate else None
    regressions, missing = gate(metrics, baseline, args.tolerance) if baseline else (None, [])
    path = write(args.out, results, scorecard(results, skipped, baseline, regressions))
    print(path.read_text())
    if missing:
        print(f"note: {len(missing)} baseline metric(s) not measured in this run", file=sys.stderr)

    if args.write_baseline:
        existing = (
            json.loads(args.write_baseline.read_text()) if args.write_baseline.exists() else {}
        )
        args.write_baseline.write_text(
            json.dumps(existing | metrics, indent=1, sort_keys=True) + "\n"
        )
        print(f"baseline updated: {args.write_baseline}", file=sys.stderr)
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
