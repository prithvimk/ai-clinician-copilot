"""Clinician-facing UI.  Run the API first, then:  uv run streamlit run src/copilot/ui/app.py"""

import json

import streamlit as st

from copilot.guardrails.checks import DISCLAIMER
from copilot.ui.client import ApiClient, ApiError

SEVERITY_ICON = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "⚪"}
PRIORITY_ICON = {"urgent": "⏱️ Urgent", "soon": "📅 Soon", "routine": "🗓️ Routine"}

st.set_page_config(page_title="AI Clinician Copilot", page_icon="🩺", layout="wide")
api = ApiClient()
state = st.session_state
state.setdefault("chat_thread", None)
state.setdefault("chat_log", [])  # [{"question", "response"}]
state.setdefault("analysis", None)  # latest AnalysisStatus dict
state.setdefault("rated", set())  # feedback already sent, by key

with st.sidebar:
    st.title("🩺 Clinician Copilot")
    st.caption(f"API: {api.base_url} — {'connected' if api.health() else '⚠️ unreachable'}")
    st.info(DISCLAIMER)
    if st.button("New conversation"):
        state.chat_thread, state.chat_log = None, []


def feedback_controls(key: str, kind: str, thread_id: str, payload: dict) -> None:
    if key in state.rated:
        st.caption("Thanks — feedback recorded.")
        return
    cols = st.columns([1, 1, 6])
    rating = "up" if cols[0].button("👍", key=f"up-{key}") else None
    rating = "down" if cols[1].button("👎", key=f"down-{key}") else rating
    correction = cols[2].text_input(
        "Correction (optional)",
        key=f"fix-{key}",
        label_visibility="collapsed",
        placeholder="What should it have said? (optional, sent with 👍/👎)",
    )
    if rating:
        try:
            api.feedback(
                kind=kind,
                thread_id=thread_id,
                rating=rating,
                correction=correction,
                payload=payload,
            )
            state.rated.add(key)
            st.rerun()
        except ApiError as e:
            st.error(str(e))


def citations_block(citations: list[dict]) -> None:
    for c in citations:
        with st.expander(f"[{c['id']}] {c['label']} · relevance {c['relevance']:.2f}"):
            st.write(c["quote"])


chat_tab, analysis_tab = st.tabs(["Guideline Q&A", "Patient analysis"])

# --- Q&A -------------------------------------------------------------------------------

with chat_tab:
    for i, turn in enumerate(state.chat_log):
        r = turn["response"]
        st.chat_message("user").write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(r["answer"])
            if r["standalone_question"] != turn["question"]:
                st.caption(f"Searched as: {r['standalone_question']}")
            if r.get("refusal_reason"):
                st.caption(f"Not answered: {r['refusal_reason'].replace('_', ' ')}")
            if r.get("guardrail_flags"):
                st.caption("Guardrails: " + ", ".join(r["guardrail_flags"]))
            citations_block(r["citations"])
            feedback_controls(
                f"chat-{r['thread_id']}-{i}",
                "chat",
                r["thread_id"],
                {
                    "question": turn["question"],
                    "answer": r["answer"],
                    "citations": [c["chunk_id"] for c in r["citations"]],
                },
            )

    if question := st.chat_input("Ask about the indexed guidelines…"):
        try:
            with st.spinner("Searching guidelines…"):
                response = api.chat(question, state.chat_thread)
            state.chat_thread = response["thread_id"]
            state.chat_log.append({"question": question, "response": response})
            st.rerun()
        except ApiError as e:
            st.error(str(e))

# --- patient analysis ------------------------------------------------------------------


def findings_block(findings: list[dict]) -> None:
    st.subheader("Findings")
    for f in findings:
        st.markdown(f"{SEVERITY_ICON[f['severity']]} **{f['title']}** — {f['detail']}")


def trace_block(trace: list[dict]) -> None:
    with st.expander(f"Agent trace ({len(trace)} steps)"):
        for i, t in enumerate(trace, 1):
            st.markdown(f"{i}. `{t['node']}` — {t['summary']}")


def recommendations_block(recs: list[dict], labels: dict[int, str]) -> None:
    for r in recs:
        cites = " ".join(f"[{c}]" for c in r["citations"])
        st.markdown(
            f"**{PRIORITY_ICON[r['priority']]}: {r['action']}** {cites}  \n{r['rationale']}"
        )
        for mc in r["medication_changes"]:
            st.caption(f"↳ {mc['change']} {mc['drug']} {mc['daily_dose']}".rstrip())
    if labels:
        with st.expander("Sources"):
            for i, label in sorted(labels.items()):
                st.markdown(f"[{i}] {label}")


def issues_block(issues: list[dict], title: str) -> None:
    if issues:
        st.warning(title + "\n\n" + "\n".join(f"- **{i['kind']}**: {i['message']}" for i in issues))


with analysis_tab:
    try:
        patients = api.patients()
    except ApiError as e:
        patients = []
        st.error(str(e))
    options = {p["patient_id"]: f"{p['patient_id']} — {p['summary'][:90]}" for p in patients}
    left, right = st.columns([4, 1])
    patient_id = left.selectbox("Patient (synthetic)", list(options), format_func=options.get)
    if right.button("Run analysis", type="primary", disabled=not patient_id):
        try:
            with st.spinner("Analysing record, retrieving guidelines, drafting and validating…"):
                state.analysis = api.analyze(patient_id)
        except ApiError as e:
            st.error(str(e))

    a = state.analysis
    if a:
        trace_block(a["trace"])
        if a["status"] == "failed":
            st.error(f"Analysis failed: {a['error']}")
            if st.button("Resume from last completed step"):
                try:
                    state.analysis = api.resume(a["thread_id"])
                    st.rerun()
                except ApiError as e:
                    st.error(str(e))

        elif a["status"] == "awaiting_review":
            req = a["review_request"]
            findings_block(req["findings"])
            st.subheader("Draft plan — awaiting your review")
            issues_block(req["issues"], "Validation could not resolve these issues:")
            recommendations_block(
                req["draft"]["recommendations"], {e["id"]: e["label"] for e in req["evidence"]}
            )
            with st.form("review"):
                decision = st.radio("Decision", ["approve", "edit", "reject"], horizontal=True)
                comment = st.text_input("Comment")
                edited = st.text_area(
                    "Recommendations (JSON, used when editing)",
                    json.dumps(req["draft"]["recommendations"], indent=2),
                    height=240,
                )
                if st.form_submit_button("Submit review"):
                    try:
                        recs = json.loads(edited) if decision == "edit" else None
                        state.analysis = api.review(a["thread_id"], decision, comment, recs)
                        st.rerun()
                    except (json.JSONDecodeError, ApiError) as e:
                        st.error(f"Review not submitted: {e}")

        elif a["status"] == "completed":
            report = a["report"]
            badge = "✅ Approved" if report["status"] == "approved" else "❌ Rejected"
            st.subheader(f"Report — {badge} · {report['confidence']} confidence")
            st.caption(report["confidence_reason"])
            st.write(report["summary"])
            issues_block(report["unresolved_issues"], "Unresolved issues (clinician override):")
            if report["actions"]:
                st.subheader("Actions")
                recommendations_block(
                    report["actions"], {r["id"]: r["label"] for r in report["references"]}
                )
            if report["monitoring"]:
                st.subheader("Monitoring")
                for m in report["monitoring"]:
                    st.markdown(
                        f"- **{m['test']}**: {m['interval']} "
                        + " ".join(f"[{c}]" for c in m["citations"])
                    )
            if report["reviewer_comment"]:
                st.caption(f"Reviewer: {report['reviewer_comment']}")
            findings_block(report["findings"])
            st.info(report["disclaimer"])
            feedback_controls(
                f"report-{a['thread_id']}",
                "analysis",
                a["thread_id"],
                {"report_status": report["status"]},
            )
