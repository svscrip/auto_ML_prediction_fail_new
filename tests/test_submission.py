"""Check the submission's links and BPMN control flow without executing it."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from src.config import CAT_FEATURES, MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
BPMN = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"


def test_card_feature_contract_matches_code():
    card = yaml.safe_load((ROOT / "docs/model-card.yaml").read_text(encoding="utf-8"))
    assert card["model_details"]["features"] == len(MODEL_FEATURES)
    assert card["model_details"]["categorical_features"] == CAT_FEATURES


def test_submission_links_resolve():
    for relative in [
        "README.md",
        "model_card.md",
        "Model_BPMN.md",
        "Model_retrain_BPMN.md",
        "docs/assignments.md",
        "docs/defense.md",
        "docs/presentation.md",
    ]:
        source = ROOT / relative
        text = source.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            assert (source.parent / target.split("#", 1)[0]).exists(), (
                relative,
                target,
            )


def _reachable(edges, start, blocked=None):
    visited = set()
    pending = [start]
    while pending:
        node = pending.pop()
        if node == blocked or node in visited:
            continue
        visited.add(node)
        pending.extend(edges.get(node, []))
    return visited


def test_bpmn_is_complete_and_cannot_publish_without_review():
    root = ET.parse(ROOT / "docs/bpmn/retraining-yanshin.bpmn").getroot()
    process = root.find(f"{BPMN}process")
    assert process is not None and process.get("isExecutable") == "false"
    nodes = {
        child.get("id"): child
        for child in process
        if child.get("id") and child.tag != f"{BPMN}sequenceFlow"
    }
    edges = {key: [] for key in nodes}
    for flow in process.findall(f"{BPMN}sequenceFlow"):
        source, target = flow.get("sourceRef"), flow.get("targetRef")
        assert source in nodes and target in nodes
        edges[source].append(target)
    for event in process.findall(f"{BPMN}boundaryEvent"):
        edges[event.get("attachedToRef")].append(event.get("id"))
    assert _reachable(edges, "Start") == set(nodes)
    for node in nodes:
        assert "End" in _reachable(edges, node), node
    without_review = _reachable(edges, "Start", blocked="Review")
    assert "Register" not in without_review
    assert "Promote" not in without_review
    assert nodes["Review"].tag == f"{BPMN}userTask"
    assert nodes["Timeout"].find(f"{BPMN}timerEventDefinition") is not None
    assert "Stop" in edges["Timeout"]
