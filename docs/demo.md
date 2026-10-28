# Demo scenario (≈ 12 minutes live, inside the 60-minute session)

Setup (before the session): index the guidelines, start the API and UI, set `ANTHROPIC_API_KEY`, open Jaeger or Langfuse in a second tab. Keep `eval/reports/scorecard.md` generated and open. Have a recorded fallback ready in case the LLM is slow.

## Act 1 — Cited Q&A (3 min)

1. Ask: *"What is the target blood pressure for adults with hypertension and diabetes?"* Show the answer, expand a citation, point at document, page and section.
2. Follow-up: *"And if they have CKD?"* Show the rewritten standalone question (history resolution).
3. Ask something out of scope: *"What dose of insulin should I take tonight?"* Show the refusal. Say why: scope guardrail, no LLM call.

## Act 2 — Patient P001, the safety loop (6 min)

P001: 58 M, type 2 diabetes, CKD 3b, hypertension, **lisinopril angioedema**, on metformin 1000 mg BD.

1. Analyse P001. Walk the findings: HbA1c, BP, falling eGFR, metformin at this eGFR.
2. Open the agent trace. Show retrieve (one query per finding) and reason.
3. Show the validation issue when the first draft misses a constraint (metformin vs eGFR, or an ACE-inhibitor given the angioedema allergy). Show the loop back and the corrected draft. Message: the model proposes, rules verify.
4. At the review gate, **edit** one recommendation, show it being re-validated, then approve.
5. Show the final report: prioritised actions, monitoring, references, disclaimer, confidence.
6. Give 👎 with a correction and show the line in the feedback JSONL.

If the model passes validation first time, say so: the loop is exercised in `tests/test_agents.py` and `agent_llm` eval.

## Act 3 — Evidence it works and runs (3 min)

1. Scorecard: agent suite at 1.0, retrieval ablation table (dense vs BM25 vs hybrid vs +rerank), refusal recall.
2. A trace in Jaeger/Langfuse: per-node latency, tokens, cost. Show that no prompt or patient text is in the span.
3. `kubectl get hpa,pdb,networkpolicy -n clinician-copilot` or the manifests, and the DR table.

## Likely questions

- *Why not let the LLM check safety?* Because contraindications are rules we can test exhaustively; the LLM cannot be gated in CI.
- *What if the corpus is wrong or stale?* Citations show source and year; manifest is versioned; re-ingest is idempotent.
- *What stops PHI reaching the model?* Pseudonymisation before every call; restored only in the output.
- *What is not done?* Thresholds are demonstrative, not a validated rule set; no real FHIR integration; golden set covers the sample pathway.
