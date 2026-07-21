"""Verifier contract tests — pure functions, [0,1], meta-driven (R1/R7)."""
from __future__ import annotations

import pytest

from gym import verifiers
from gym.verifiers import (anchor_recall, final_number, numeric_cloze,
                           numeric_tolerance, ontology_qa, sandbox_tests, sparql_exec)


def test_registry():
    assert callable(verifiers.get("numeric_cloze"))
    with pytest.raises(KeyError):
        verifiers.get("nope")


def test_numeric_cloze():
    assert numeric_cloze.verify("", "45 °C in the supply line", {"value": 45}) == 1.0
    assert numeric_cloze.verify("", " 45.0 degrees", {"value": 45}) == 1.0
    assert numeric_cloze.verify("", "44 °C", {"value": 45}) == 0.0
    assert numeric_cloze.verify("", "no number here", {"value": 45}) == 0.0
    assert numeric_cloze.verify("", "about 1,600 times", {"value": 1600}) == 1.0
    # only the first number in the opening window counts
    assert numeric_cloze.verify("", "3 things: the answer is 45", {"value": 45}) == 0.0


def test_final_number():
    assert final_number.verify("", "so #### 42", {"gold": "blah #### 42"}) == 1.0
    assert final_number.verify("", "so #### 41", {"gold": "blah #### 42"}) == 0.1
    assert final_number.verify("", "no answer", {"gold": "#### 42"}) == 0.0
    assert final_number.verify("", "it is 42", {"gold": "#### 42"}) == 1.0  # last number


def test_numeric_tolerance():
    assert numeric_tolerance.verify("", "roughly 9.8", {"value": 10, "atol": 0.5}) == 1.0
    assert numeric_tolerance.verify("", "roughly 9.4", {"value": 10, "atol": 0.5}) == 0.0
    assert numeric_tolerance.verify("", "102", {"value": 100, "rtol": 0.02}) == 1.0
    assert numeric_tolerance.verify("", "7 then 10", {"value": 7, "which": "first"}) == 1.0
    assert numeric_tolerance.verify("", "nothing", {"value": 1}) == 0.0


def test_ontology_qa():
    assert ontology_qa.verify("", "It is a Damper.", {"gold": "type:Damper|Actuator"}) == 1.0
    assert ontology_qa.verify("", "A valve", {"gold": "type:Damper"}) == 0.0
    assert ontology_qa.verify("", "there are 7", {"gold": "count:7"}) == 1.0
    assert ontology_qa.verify("", "6 or 7? 6", {"gold": "count:7"}) == 0.0
    assert ontology_qa.verify("", "connects to VLV-1 and PMP-2",
                              {"gold": "conns:VLV-1|PMP-2"}) == 1.0
    assert ontology_qa.verify("", "connects to VLV-1",
                              {"gold": "conns:VLV-1|PMP-2"}) == 0.5


def test_anchor_recall():
    meta = {"anchors": ["21.5 °C", "AHU-01", "supply damper", "07:00"]}
    resp = "Set AHU-01 to 21.5 °C at startup."
    assert anchor_recall.verify("", resp, meta) == 0.5
    assert anchor_recall.verify("", "", meta) == 0.0
    assert anchor_recall.verify("", "anything", {"anchors": []}) == 0.0


TTL = """
@prefix ex: <http://example.org/> .
ex:vlv1 a ex:Valve . ex:vlv2 a ex:Valve . ex:pmp1 a ex:Pump .
"""


def test_sparql_exec():
    meta = {"ttl": TTL,
            "gold_query": "SELECT ?s WHERE { ?s a <http://example.org/Valve> }"}
    good = "```sparql\nSELECT ?x WHERE { ?x a <http://example.org/Valve> }\n```"
    assert sparql_exec.verify("", good, meta) == 1.0
    broad = "SELECT ?x WHERE { ?x a ?t }"
    assert 0.0 < sparql_exec.verify("", broad, meta) < 1.0   # recall 1, precision 2/3
    assert sparql_exec.verify("", "no query at all", meta) == 0.0
    assert sparql_exec.verify("", "SELECT ?x WHERE { broken", meta) == 0.0


def test_sandbox_tests():
    meta = {"tests": "import solution\nassert solution.add(1, 2) == 3\n", "timeout_s": 15}
    ok = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert sandbox_tests.verify("", ok, meta) == 1.0
    assert sandbox_tests.verify("", bad, meta) == 0.0
    assert sandbox_tests.verify("", ok, {"tests": ""}) == 0.0
