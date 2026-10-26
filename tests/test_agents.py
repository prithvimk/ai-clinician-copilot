import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from copilot.agents.patient import Lab, Patient, daily_dose_mg, load_patient
from copilot.agents.rules import analyse_patient
from copilot.agents.safety import check_medication, drug_classes
from copilot.agents.service import AnalysisService, ReviewNotPending, get_analysis_service
from copilot.agents.workflow import AnalysisWorkflow, Draft, ReviewDecision
from copilot.api.app import app
from tests.fakes import FakeLLM, StubRetriever

P001_SERIOUS = {
    "hba1c-above-target",
    "bp-above-target",
    "egfr-reduced",
    "albuminuria",
    "med-metformin_egfr-metformin",
}

# --- rules -----------------------------------------------------------------------------


def test_p001_findings():
    findings = {f.id: f for f in analyse_patient(load_patient("P001"))}
    assert {fid for fid, f in findings.items() if f.severity == "high"} == P001_SERIOUS
    assert "fell 7 in 6 months" in findings["egfr-reduced"].detail
    assert "2000 mg/day" in findings["med-metformin_egfr-metformin"].detail
    assert "Amlodipine" not in findings["hba1c-above-target"].detail  # not glucose-lowering
    assert {"ldl-raised", "med-sulfonylurea_ckd-gliclazide", "potassium-raised"} <= set(findings)


def test_p002_and_p003_findings():
    p002 = {f.id: f.severity for f in analyse_patient(load_patient("P002"))}
    assert p002 == {
        "bp-above-target": "high",
        "hba1c-prediabetes": "low",
        "cholesterol-raised": "low",
    }
    p003 = {f.id: f.severity for f in analyse_patient(load_patient("P003"))}
    assert p003 == {"hba1c-diabetes-range": "high"}


def test_findings_sorted_by_severity():
    order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
    severities = [order[f.severity] for f in analyse_patient(load_patient("P001"))]
    assert severities == sorted(severities)


# --- safety ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "mg"),
    [("1000 mg twice daily", 2000), ("500mg", 500), ("1 g bd", 2000), ("as directed", None)],
)
def test_daily_dose(text, mg):
    assert daily_dose_mg(text) == mg


def test_drug_classes():
    assert drug_classes("Ramipril") == {"ace_inhibitor"}
    assert drug_classes("an ACE inhibitor") == {"ace_inhibitor"}
    assert drug_classes("Dapagliflozin 10 mg") == {"sglt2_inhibitor"}
    assert drug_classes("paracetamol") == set()


def _rules(patient, drug, change, dose=""):
    return {(i.rule, i.severity) for i in check_medication(patient, drug, change, dose)}


def test_allergy_class_and_cross_reactivity():
    p = load_patient("P001")  # lisinopril angioedema
    assert ("allergy", "critical") in _rules(p, "Ramipril", "start")
    assert ("allergy_cross_reactivity", "high") in _rules(p, "Losartan", "start")
    assert _rules(p, "Lisinopril", "stop") == set()


def test_metformin_by_egfr():
    p = load_patient("P001")  # eGFR 34, metformin 1000 mg twice daily
    assert ("metformin_egfr", "high") in _rules(p, "Metformin", "continue")
    assert _rules(p, "Metformin", "reduce", "1000 mg") == set()
    assert ("metformin_egfr", "high") in _rules(p, "Metformin", "continue", "1500 mg")

    p.labs.append(Lab(date="2027-01-01", test="eGFR", value=25))
    assert ("metformin_egfr", "critical") in _rules(p, "Metformin", "reduce", "500 mg")


def test_sglt2_allowed_at_egfr_34():
    assert _rules(load_patient("P001"), "Empagliflozin", "start", "10 mg") == set()


# --- workflow --------------------------------------------------------------------------


def rec(finding_ids, action, changes=(), citations=(1,), priority="soon"):
    return {
        "finding_ids": list(finding_ids),
        "action": action,
        "rationale": "Per guideline.",
        "priority": priority,
        "medication_changes": [
            {"drug": d, "change": c, "daily_dose": dose} for d, c, dose in changes
        ],
        "citations": list(citations),
    }


MONITORING = [{"test": "eGFR", "interval": "every 3-6 months", "citations": [1]}]
SGLT2 = rec(
    ["hba1c-above-target", "egfr-reduced", "albuminuria"],
    "Start an SGLT2 inhibitor",
    [("Empagliflozin", "start", "10 mg")],
)
BP = rec(["bp-above-target"], "Intensify antihypertensive therapy", [("Indapamide", "start", "")])


def draft(*metformin_rec, unsupported=()):
    return Draft.model_validate(
        {
            "recommendations": [SGLT2, BP, *metformin_rec],
            "monitoring": MONITORING,
            "unsupported_findings": list(unsupported),
        }
    )


UNSAFE = draft(
    rec(
        ["med-metformin_egfr-metformin"],
        "Continue metformin",
        [("Metformin", "continue", "2000 mg")],
    )
)
SAFE = draft(
    rec(["med-metformin_egfr-metformin"], "Reduce metformin", [("Metformin", "reduce", "1000 mg")])
)


def service_with(llm, retriever=None, max_revisions=2) -> tuple[AnalysisService, StubRetriever]:
    retriever = retriever or StubRetriever(
        "Reduce metformin to max 1000 mg/day at eGFR 30-44.",
        "Start an SGLT2 inhibitor when eGFR >= 20.",
        "Review BP monthly until at target.",
    )
    workflow = AnalysisWorkflow(
        retriever, llm, InMemorySaver(), min_relevance=0.2, max_revisions=max_revisions
    )
    return AnalysisService(workflow), retriever


def test_metformin_validation_failure_self_corrects_then_pauses_for_review():
    llm = FakeLLM(UNSAFE, SAFE)
    service, _ = service_with(llm)

    paused = service.start(load_patient("P001"), "t1")
    assert paused.status == "awaiting_review"
    assert [t["node"] for t in paused.trace] == [
        "analyze",
        "retrieve",
        "reason",
        "validate",
        "reason",
        "validate",
    ]
    assert "exceeds the 1000 mg/day maximum" in paused.trace[3]["summary"]
    revision_prompt = llm.calls[1]["messages"][0]["content"]
    assert "<validation_issues>" in revision_prompt and "Metformin" in revision_prompt
    assert paused.review_request["issues"] == []

    done = service.review("t1", ReviewDecision(decision="approve", comment="Agree"))
    assert done.status == "completed"
    report = done.report
    assert report["status"] == "approved" and report["revisions"] == 1
    assert report["confidence"] == "medium" and report["unresolved_issues"] == []
    assert any(a["action"] == "Reduce metformin" for a in report["actions"])
    assert [r["id"] for r in report["references"]] == [1]
    assert report["reviewer_comment"] == "Agree" and "Decision support only" in report["disclaimer"]


def test_unknown_citation_triggers_revision():
    bad = draft(
        rec(
            ["med-metformin_egfr-metformin"],
            "Reduce metformin",
            [("Metformin", "reduce", "1000 mg")],
            citations=(42,),
        )
    )
    service, _ = service_with(FakeLLM(bad, SAFE))
    paused = service.start(load_patient("P001"), "t")
    assert "unknown source ids [42]" in paused.trace[3]["summary"]
    assert paused.review_request["issues"] == []


def test_finding_without_evidence_reretrieves_with_broader_query():
    # Nothing found for the albuminuria query at first, so the draft can't address it.
    retriever = StubRetriever("Kidney guidance.", miss=lambda q: q.startswith("albuminuria"))
    no_albuminuria = draft(SAFE.recommendations[2].model_dump())
    no_albuminuria.recommendations[0].finding_ids = ["hba1c-above-target", "egfr-reduced"]
    service, _ = service_with(FakeLLM(no_albuminuria, SAFE), retriever)

    paused = service.start(load_patient("P001"), "t")
    nodes = [t["node"] for t in paused.trace]
    assert nodes == ["analyze", "retrieve", "reason", "validate", "retrieve", "reason", "validate"]
    assert retriever.queries[-1].startswith("Severely increased albuminuria")
    assert paused.status == "awaiting_review" and paused.review_request["issues"] == []


def test_gives_up_after_max_revisions_and_flags_issues():
    service, _ = service_with(FakeLLM(UNSAFE, UNSAFE, UNSAFE))
    paused = service.start(load_patient("P001"), "t")
    assert [t["node"] for t in paused.trace].count("reason") == 3
    assert paused.review_request["issues"][0]["kind"] == "safety"

    report = service.review("t", ReviewDecision(decision="approve")).report
    assert report["confidence"] == "low"
    assert report["unresolved_issues"][0]["kind"] == "safety"


def test_clinician_edit_is_rechecked():
    service, _ = service_with(FakeLLM(SAFE))
    service.start(load_patient("P001"), "t")
    edited = [
        *SAFE.recommendations,
        Draft.model_validate(
            {
                "recommendations": [
                    rec(["bp-above-target"], "Add ramipril", [("Ramipril", "start", "2.5 mg")])
                ],
                "monitoring": [],
                "unsupported_findings": [],
            }
        ).recommendations[0],
    ]

    report = service.review("t", ReviewDecision(decision="edit", recommendations=edited)).report
    assert report["actions"][-1]["action"] == "Add ramipril"
    assert "allergic to Lisinopril" in report["unresolved_issues"][0]["message"]
    assert report["confidence"] == "low"


def test_reject():
    service, _ = service_with(FakeLLM(SAFE))
    service.start(load_patient("P001"), "t")
    report = service.review("t", ReviewDecision(decision="reject", comment="Seen today")).report
    assert report["status"] == "rejected" and report["actions"] == []
    assert report["findings"]  # findings are still reported

    with pytest.raises(ReviewNotPending):
        service.review("t", ReviewDecision(decision="approve"))


def test_resume_after_failure_skips_completed_steps():
    llm = FakeLLM(RuntimeError("provider outage"))
    service, retriever = service_with(llm)

    failed = service.start(load_patient("P001"), "t")
    assert failed.status == "failed" and "provider outage" in failed.error
    queries_before = len(retriever.queries)

    llm.replies = [SAFE]
    resumed = service.resume("t")
    assert resumed.status == "awaiting_review"
    assert len(retriever.queries) == queries_before  # analyze/retrieve were not re-run
    assert [t["node"] for t in resumed.trace] == ["analyze", "retrieve", "reason", "validate"]


def test_unknown_thread():
    service, _ = service_with(FakeLLM())
    assert service.status("missing").status == "not_found"


# --- API -------------------------------------------------------------------------------


def test_analyze_endpoints():
    service, _ = service_with(FakeLLM(SAFE))
    app.dependency_overrides[get_analysis_service] = lambda: service
    try:
        client = TestClient(app)
        paused = client.post("/analyze", json={"patient_id": "P001"}).json()
        assert paused["status"] == "awaiting_review"
        thread_id = paused["thread_id"]
        assert client.get(f"/analyze/{thread_id}").json()["status"] == "awaiting_review"
        assert client.post(f"/analyze/{thread_id}/resume").status_code == 409

        done = client.post(f"/analyze/{thread_id}/review", json={"decision": "approve"}).json()
        assert done["status"] == "completed" and done["report"]["status"] == "approved"
        assert (
            client.post(f"/analyze/{thread_id}/review", json={"decision": "approve"}).status_code
            == 409
        )

        assert client.post("/analyze", json={"patient_id": "NOPE"}).status_code == 404
        assert client.post("/analyze", json={}).status_code == 422
        assert client.get("/analyze/missing").status_code == 404

        inline = Patient(patient_id="X1", problems=["Hypertension"]).model_dump(mode="json")
        service_inline, _ = service_with(
            FakeLLM({"recommendations": [], "monitoring": [], "unsupported_findings": []})
        )
        app.dependency_overrides[get_analysis_service] = lambda: service_inline
        assert (
            client.post("/analyze", json={"patient": inline}).json()["status"] == "awaiting_review"
        )
    finally:
        app.dependency_overrides.clear()
