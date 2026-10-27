import sys

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from mcp import StdioServerParameters

from copilot.agents.interactions import check_interactions
from copilot.agents.patient import load_patient
from copilot.agents.service import build_analysis_service
from copilot.agents.workflow import Draft, ReviewDecision
from copilot.mcp_server.client import McpRetriever, McpToolbox, ToolCallError
from copilot.mcp_server.fhir import from_bundle, to_bundle
from copilot.mcp_server.server import build_server
from tests.fakes import FakeLLM, StubRetriever
from tests.test_agents import SAFE, rec


@pytest.fixture(scope="module")
def toolbox():
    tb = McpToolbox(
        build_server(lambda: StubRetriever("Reduce metformin at eGFR 30-44.", "SGLT2."))
    )
    yield tb
    tb.close()


@pytest.mark.parametrize("pid", ["P001", "P002", "P003"])
def test_fhir_round_trip(pid):
    patient = load_patient(pid)
    bundle = to_bundle(patient)
    types = {e["resource"]["resourceType"] for e in bundle["entry"]}
    assert {"Patient", "Condition", "Observation", "DocumentReference"} <= types
    assert from_bundle(bundle) == patient


def test_fhir_uses_loinc_codes():
    bundle = to_bundle(load_patient("P001"))
    codes = {
        c["code"]
        for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Observation"
        for c in e["resource"]["code"]["coding"]
    }
    assert {"4548-4", "33914-3", "8480-6"} <= codes  # HbA1c, eGFR, systolic BP


def test_tools_listed(toolbox):
    assert set(toolbox.list_tools()) == {
        "search_guidelines",
        "get_patient_record",
        "check_drug_interaction",
    }


def test_get_patient_record(toolbox):
    assert toolbox.get_patient("P001") == load_patient("P001")
    with pytest.raises(ToolCallError, match="Unknown patient"):
        toolbox.get_patient("NOPE")


def test_search_guidelines(toolbox):
    hits = McpRetriever(toolbox).retrieve("metformin", k=1)
    assert len(hits) == 1 and hits[0].chunk.text.startswith("Reduce metformin")
    with pytest.raises(ToolCallError):
        toolbox.search_guidelines("  ")


def test_check_drug_interaction(toolbox):
    interactions, issues = toolbox.check_drugs(["Ramipril", "Losartan", "Spironolactone"], "P001")
    assert {i.drugs for i in interactions} == {
        ("Ramipril", "Losartan"),
        ("Ramipril", "Spironolactone"),
        ("Losartan", "Spironolactone"),
    }
    assert {(i.drug, i.rule) for i in issues} == {
        ("Ramipril", "allergy"),
        ("Losartan", "allergy_cross_reactivity"),
    }
    assert toolbox.check_drugs(["Amlodipine", "Metformin"]) == ([], [])


def test_interactions_are_deduplicated():
    assert len(check_interactions(["Ramipril", "Losartan", "Ramipril"])) == 1


def test_workflow_through_mcp_flags_introduced_interaction(toolbox):
    """Patient, retrieval and interaction checks all go through the MCP tools."""
    with_spironolactone = Draft.model_validate(
        {
            "recommendations": [
                *[r.model_dump() for r in SAFE.recommendations],
                rec(["albuminuria"], "Add spironolactone", [("Spironolactone", "start", "25 mg")]),
            ],
            "monitoring": SAFE.model_dump()["monitoring"],
            "unsupported_findings": [],
        }
    )
    # Start an ARB alongside: ARB + MRA is flagged; the revised draft drops spironolactone.
    with_spironolactone.recommendations[1].medication_changes[0].drug = "Candesartan"
    service = build_analysis_service(
        McpRetriever(toolbox),
        FakeLLM(with_spironolactone, SAFE),
        InMemorySaver(),
        patient_source=toolbox.get_patient,
        interaction_checker=lambda drugs: toolbox.check_drugs(drugs)[0],
    )
    paused = service.start(service.load_patient("P001"), "t")
    validate = [t["summary"] for t in paused.trace if t["node"] == "validate"]
    assert "Candesartan + Spironolactone" in validate[0]
    assert validate[1] == "passed"
    assert service.review("t", ReviewDecision(decision="approve")).status == "completed"


def test_stdio_transport():
    tb = McpToolbox(
        StdioServerParameters(command=sys.executable, args=["-m", "copilot.mcp_server"])
    )
    try:
        assert tb.get_patient("P002").problems == ["Essential hypertension"]
    finally:
        tb.close()
