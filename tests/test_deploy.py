"""Structural checks on the deployment manifests (schema validation runs in CI via
kubeconform; these catch wiring mistakes a schema can't)."""

import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from copilot.config import PROJECT_ROOT, Settings

DEPLOY = PROJECT_ROOT / "deploy"
BASE = DEPLOY / "k8s" / "base"
WORKLOADS = {"Deployment", "StatefulSet", "Job", "CronJob"}
APP_IMAGE = "clinician-copilot"  # rewritten by kustomize `images:`
NON_SETTINGS_ENV = {"HF_HUB_OFFLINE", "COPILOT_API_URL", "HOME"}


def load_docs(directory: Path) -> list[dict]:
    docs = []
    for path in sorted(directory.glob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        docs += [d for d in yaml.safe_load_all(path.read_text()) if d]
    return docs


DOCS = load_docs(BASE)


def of_kind(*kinds):
    return [d for d in DOCS if d["kind"] in kinds]


def pod_spec(doc: dict) -> dict:
    spec = doc["spec"]
    if doc["kind"] == "CronJob":
        spec = spec["jobTemplate"]["spec"]
    return spec["template"]


def containers(doc: dict) -> list[dict]:
    spec = pod_spec(doc)["spec"]
    return spec.get("initContainers", []) + spec["containers"]


def labels_match(selector: dict, labels: dict) -> bool:
    return all(labels.get(k) == v for k, v in selector.items())


def test_kustomization_lists_every_manifest():
    kustomization = yaml.safe_load((BASE / "kustomization.yaml").read_text())
    files = {p.name for p in BASE.glob("*.yaml")} - {"kustomization.yaml"}
    assert set(kustomization["resources"]) == files


@pytest.mark.parametrize("doc", of_kind(*WORKLOADS), ids=lambda d: d["metadata"]["name"])
def test_workloads_are_hardened_and_sized(doc):
    pod = pod_spec(doc)["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert pod.get("automountServiceAccountToken") is False
    for c in containers(doc):
        res = c["resources"]
        assert {"cpu", "memory"} <= set(res["requests"]) and {"cpu", "memory"} <= set(res["limits"])
        sc = c["securityContext"]
        assert sc["allowPrivilegeEscalation"] is False and sc["capabilities"]["drop"] == ["ALL"]
        image = c["image"]
        assert image == APP_IMAGE or re.search(r":v?\d", image), f"unpinned image {image}"
        if image == APP_IMAGE:
            assert sc["readOnlyRootFilesystem"] is True
            mounts = {m["mountPath"] for m in c.get("volumeMounts", [])}
            assert c["name"] == "corpus-manifest" or "/tmp" in mounts
    if doc["kind"] in ("Deployment", "StatefulSet"):
        for c in pod["containers"]:
            assert "readinessProbe" in c and "livenessProbe" in c, c["name"]


@pytest.mark.parametrize(
    "doc", of_kind("Deployment", "StatefulSet"), ids=lambda d: d["metadata"]["name"]
)
def test_selectors_match_pod_labels(doc):
    assert labels_match(doc["spec"]["selector"]["matchLabels"], pod_spec(doc)["metadata"]["labels"])


def test_every_deployment_has_a_pdb_and_api_autoscales():
    pdbs = of_kind("PodDisruptionBudget")
    for d in of_kind("Deployment"):
        labels = pod_spec(d)["metadata"]["labels"]
        assert any(labels_match(p["spec"]["selector"]["matchLabels"], labels) for p in pdbs), d[
            "metadata"
        ]["name"]
    deployments = {d["metadata"]["name"] for d in of_kind("Deployment")}
    [hpa] = of_kind("HorizontalPodAutoscaler")
    assert hpa["spec"]["scaleTargetRef"]["name"] in deployments
    assert hpa["spec"]["minReplicas"] >= 2


def test_services_and_ingress_resolve():
    pods = {d["metadata"]["name"]: d for d in of_kind("Deployment", "StatefulSet")}
    services = {s["metadata"]["name"]: s for s in of_kind("Service")}
    for svc in services.values():
        targets = [
            d
            for d in pods.values()
            if labels_match(svc["spec"]["selector"], pod_spec(d)["metadata"]["labels"])
        ]
        assert targets, f"service {svc['metadata']['name']} selects nothing"
        port_names = {
            p.get("name")
            for d in targets
            for c in pod_spec(d)["spec"]["containers"]
            for p in c["ports"]
        }
        for port in svc["spec"]["ports"]:
            assert port["targetPort"] in port_names
    for ing in of_kind("Ingress"):
        for rule in ing["spec"]["rules"]:
            for path in rule["http"]["paths"]:
                assert path["backend"]["service"]["name"] in services
    for sm in of_kind("ServiceMonitor"):
        [svc] = [
            s
            for s in services.values()
            if labels_match(sm["spec"]["selector"]["matchLabels"], s["metadata"].get("labels", {}))
        ]
        assert {e["port"] for e in sm["spec"]["endpoints"]} <= {
            p["name"] for p in svc["spec"]["ports"]
        }


def test_network_policies_cover_every_pod():
    policies = of_kind("NetworkPolicy")
    assert any(
        p["spec"]["podSelector"] == {} and set(p["spec"]["policyTypes"]) == {"Ingress", "Egress"}
        for p in policies
    )
    specific = [p for p in policies if p["spec"]["podSelector"].get("matchLabels")]
    for d in of_kind(*WORKLOADS):
        labels = pod_spec(d)["metadata"]["labels"]
        assert any(
            labels_match(p["spec"]["podSelector"]["matchLabels"], labels) for p in specific
        ), d["metadata"]["name"]


def test_config_keys_are_real_settings():
    [cm] = of_kind("ConfigMap")
    fields = {f.upper() for f in Settings.model_fields}
    assert set(cm["data"]) - NON_SETTINGS_ENV <= fields
    for d in of_kind(*WORKLOADS):
        for c in (c for c in containers(d) if c["image"] == APP_IMAGE):
            for env in c.get("env", []):
                assert env["name"] in fields | NON_SETTINGS_ENV, env["name"]


def test_compose_env_names_are_real_settings():
    compose = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())
    fields = {f.upper() for f in Settings.model_fields}
    for name, service in compose["services"].items():
        if service.get("image", "").startswith("${COPILOT_IMAGE"):
            assert set(service.get("environment", {})) <= fields | NON_SETTINGS_ENV, name


@pytest.mark.skipif(not shutil.which("docker"), reason="docker CLI not installed")
def test_compose_file_validates():
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(DEPLOY / "docker-compose.yml"),
            "--profile",
            "tools",
            "config",
            "--quiet",
        ],
        check=True,
    )


def test_dockerfile_copy_sources_exist():
    for line in (DEPLOY / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY") and "--from=" not in line:
            *sources, _dest = line.split()[1:]
            for src in (s for s in sources if not s.startswith("--")):
                assert (PROJECT_ROOT / src).exists(), src


def test_prod_overlay_patches_existing_resources():
    overlay = yaml.safe_load(
        (DEPLOY / "k8s" / "overlays" / "prod" / "kustomization.yaml").read_text()
    )
    names = {(d["kind"], d["metadata"]["name"]) for d in DOCS}
    for patch in overlay["patches"]:
        assert (patch["target"]["kind"], patch["target"]["name"]) in names


def test_snapshot_retention():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "snap", PROJECT_ROOT / "scripts" / "qdrant_snapshot.py"
    )
    snap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snap)
    snaps = [
        SimpleNamespace(name=f"s{i}", creation_time=f"2026-10-0{i}T02:30:00")
        for i in (3, 1, 2, 5, 4)
    ]
    assert snap.to_delete(snaps, keep=3) == ["s2", "s1"]
    assert snap.to_delete(snaps, keep=10) == []
