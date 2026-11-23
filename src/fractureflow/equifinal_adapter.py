# -*- coding: utf-8 -*-
"""Adapter: frozen observation-family manifests -> unified finite worlds.

Single conversion layer between the two frozen candidate manifests of
TO-W161-02 v2 P1 and the finite-world interface of
:mod:fractureflow.completion_certificate.

The module deliberately does **not** re-implement answer_finite,
equivalent_set, decide_equivalent_set or rank_next_observations.
The manifest kernel partition used by compute_three_state is obtained by
calling the core equivalent_set with the full declared action table observed
for one state, which is the manifest equivalence cell by definition.

Contract source (frozen, hashed in the same directory):
results/obs_equiv_20261024/w161_02_completion_certificate_20261103/p1_equifinal_loop/prereg.json
sections unified_schema and adapter_contract.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .completion_certificate import equivalent_set

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_ID = "fractureflow.equifinal.unified.v1"

ADAPTER_W38 = "w38_full_history"
ADAPTER_REAL = "real_trace_signature_by_window"
ADAPTERS = (ADAPTER_W38, ADAPTER_REAL)

DEC_CONNECTED = "DEC_CONNECTED"
DEC_DISCONNECTED = "DEC_DISCONNECTED"
UNDECIDABLE_BY_DECLARED_ACTIONS = "UNDECIDABLE_BY_DECLARED_ACTIONS"
THREE_STATES = (DEC_CONNECTED, DEC_DISCONNECTED, UNDECIDABLE_BY_DECLARED_ACTIONS)

#: Adapter token -> token used by the manifests' own truth_3state field.
MANIFEST_THREE_STATE = {
    DEC_CONNECTED: "DEC_CONNECTED",
    DEC_DISCONNECTED: "DEC_DISCONNECTED",
    UNDECIDABLE_BY_DECLARED_ACTIONS: "UNDECIDABLE",
}

BOUNDARY = {
    ADAPTER_W38: {
        "domain": "SYNTHETIC_DECLARED_DOMAIN_ONLY",
        "is_field_evidence": False,
        "physical_quantities_allowed": False,
    },
    ADAPTER_REAL: {
        "domain": "REAL_TRACE_TOPOLOGY_2D5",
        "is_field_evidence": False,
        "physical_quantities_allowed": False,
    },
}
BOUNDARY_DOMAINS = tuple(spec["domain"] for spec in BOUNDARY.values())

FIDELITY_CODE = "ADAPTER_FIDELITY_FAIL"

_MANIFEST_FIELDS = (
    "schema", "problem_id", "source", "states", "labels", "priors",
    "actions", "observed", "boundary",
)


class SchemaError(Exception):
    """Contract violation with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = str(code)
        self.message = str(message)


class FidelityFailure(SystemExit):
    """Hard stop when the computed three-state disagrees with manifest truth."""

    code = FIDELITY_CODE

    def __init__(self, message: str) -> None:
        super().__init__("%s: %s" % (self.code, message))
        self.message = str(message)


def _fail(code: str, message: str) -> None:
    raise SchemaError(code, message)


def sha256_file(path: Any) -> str:
    """SHA-256 of a file, read in binary chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Any) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return candidate.as_posix()


def _fraction_token(value: Any, *, code: str, nonpositive_code: str, where: str):
    """Parse a declared fraction and return (Fraction, canonical string).

    Floats are rejected: the frozen schema stores fractions as strings so no
    binary round-trip can silently move a cost or a prior.
    """
    if isinstance(value, bool):
        _fail(code, "%s must be a fraction string, got a bool" % where)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            _fail(code, "%s is an empty fraction string" % where)
        try:
            parsed = Fraction(token)
        except (ValueError, ZeroDivisionError):
            _fail(code, "%s is not a parseable fraction: %r" % (where, value))
        if parsed <= 0:
            _fail(nonpositive_code, "%s must be positive, got %r" % (where, value))
        return parsed, value
    if isinstance(value, int):
        if value <= 0:
            _fail(nonpositive_code, "%s must be positive, got %r" % (where, value))
        return Fraction(value), str(value)
    _fail(code, "%s must be a fraction string, got %s" % (where, type(value).__name__))


def _unit_query(unit: Mapping[str, Any], unit_id: str):
    """Validate unit.query / unit.prior and return states, labels, priors."""
    if not isinstance(unit, Mapping):
        _fail("UNIT_NOT_FOUND", "unit %r is not an object" % (unit_id,))
    declared = unit.get("unit_id", unit_id)
    if declared != unit_id:
        _fail("UNIT_ID_MISMATCH", "unit key %r carries unit_id %r" % (unit_id, declared))
    query = unit.get("query")
    prior = unit.get("prior")
    if not isinstance(query, Mapping) or not query:
        _fail("EMPTY_STATES", "unit %r has no query states" % (unit_id,))
    states = list(query.keys())
    if not all(isinstance(state, str) for state in states):
        _fail("INVALID_STATE_ID", "unit %r has a non-string state id" % (unit_id,))
    if len(set(states)) != len(states):
        _fail("DUPLICATE_STATE", "unit %r declares duplicate states" % (unit_id,))
    if not isinstance(prior, Mapping):
        _fail("QUERY_PRIOR_KEY_MISMATCH", "unit %r has no prior object" % (unit_id,))
    if list(prior.keys()) != states:
        _fail("QUERY_PRIOR_KEY_MISMATCH",
              "unit %r prior keys/order differ from query keys: %r vs %r"
              % (unit_id, list(prior.keys()), states))
    labels = {}
    for state in states:
        raw = query[state]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw not in (0, 1):
            _fail("INVALID_LABEL", "unit %r state %r label %r is not 0/1"
                  % (unit_id, state, raw))
        labels[state] = int(raw)
    priors = {}
    total = Fraction(0)
    for state in states:
        parsed, token = _fraction_token(prior[state], code="INVALID_PRIOR",
                                        nonpositive_code="INVALID_PRIOR",
                                        where="unit %r prior[%r]" % (unit_id, state))
        priors[state] = token
        total += parsed
    if total != 1:
        _fail("PRIOR_NOT_NORMALISED", "unit %r priors sum to %s, expected 1"
              % (unit_id, total))
    return states, labels, priors


def _declared_costs(window_costs: Any, where: str):
    """Validate a window cost table and return (action_order, token_by_action)."""
    if not isinstance(window_costs, Mapping) or not window_costs:
        _fail("EMPTY_ACTIONS", "%s has no declared windows" % where)
    tokens = {}
    for action_id, value in window_costs.items():
        if not isinstance(action_id, str) or not action_id:
            _fail("INVALID_ACTION_ID", "%s has a non-string window id" % where)
        _parsed, token = _fraction_token(
            value, code="INVALID_COST", nonpositive_code="NONPOSITIVE_COST",
            where="%s cost[%s]" % (where, action_id))
        tokens[action_id] = token
    return sorted(tokens), tokens


def _assemble(adapter: str, family_id: str, unit_id: str, states, labels, priors,
              actions, source, truth_3state) -> dict:
    source_block = {
        "manifest": None,
        "sha256": None,
        "family_id": family_id,
        "unit_id": unit_id,
        "adapter": adapter,
    }
    if source:
        for key, value in source.items():
            if key in source_block and value is not None:
                source_block[key] = value
    unified = {
        "schema": SCHEMA_ID,
        "problem_id": "%s__%s" % (family_id, unit_id),
        "source": source_block,
        "states": list(states),
        "labels": dict(labels),
        "priors": dict(priors),
        "actions": actions,
        "observed": {},
        "boundary": dict(BOUNDARY[adapter]),
    }
    _check_fidelity(unified, truth_3state, unit_id)
    return unified


def _check_fidelity(unified, truth_3state, unit_id: str) -> None:
    """Compare the computed three-state to the manifest truth, case by case."""
    if not isinstance(truth_3state, Mapping):
        _fail("MISSING_TRUTH_3STATE",
              "unit %r has no manifest truth_3state to check against" % (unit_id,))
    computed = compute_three_state(unified)
    bad = []
    for state in unified["states"]:
        if state not in truth_3state:
            _fail("MISSING_TRUTH_3STATE",
                  "unit %r state %r has no manifest truth_3state" % (unit_id, state))
        expected = MANIFEST_THREE_STATE[computed[state]]
        actual = truth_3state[state]
        if actual != expected:
            bad.append((state, computed[state], actual, expected))
    if bad:
        head = "; ".join("%s computed=%s manifest=%s expected_manifest_token=%s"
                         % row for row in bad[:4])
        raise FidelityFailure(
            "unit %r three-state disagrees with manifest truth on %d/%d cases: %s"
            % (unit_id, len(bad), len(unified["states"]), head))


def _full_case_index(doc, unit_id: str, adapter: str) -> dict:
    """Index the manifest cases that carry the full declared history for a unit."""
    index = {}
    for case in doc.get("cases") or ():
        if not isinstance(case, Mapping):
            _fail("INVALID_CASE", "manifest case is not an object")
        if case.get("unit_id") != unit_id:
            continue
        if adapter == ADAPTER_W38 and case.get("endpoint") != "full":
            continue
        position = case.get("position")
        if not isinstance(position, str):
            _fail("INVALID_CASE", "unit %r has a case without a string position"
                  % (unit_id,))
        if position in index:
            _fail("DUPLICATE_FULL_CASE",
                  "unit %r has more than one case for position %r" % (unit_id, position))
        index[position] = case
    return index


def adapt_w38_manifest(doc, unit_id: str, *, source=None) -> dict:
    """Convert one W38 w38_full_history unit to the unified schema."""
    units = doc.get("units")
    if not isinstance(units, Mapping) or unit_id not in units:
        _fail("UNIT_NOT_FOUND", "unit %r is absent from doc['units']" % (unit_id,))
    unit = units[unit_id]
    states, labels, priors = _unit_query(unit, unit_id)

    cost_block = doc.get("cost")
    if not isinstance(cost_block, Mapping) or "window_costs" not in cost_block:
        _fail("MISSING_WINDOW_COSTS", "doc['cost']['window_costs'] is absent")
    action_order, cost_tokens = _declared_costs(
        cost_block["window_costs"], "doc['cost']['window_costs']")

    cases = _full_case_index(doc, unit_id, ADAPTER_W38)
    if set(cases) != set(states):
        missing = sorted(set(states) - set(cases))
        extra = sorted(set(cases) - set(states))
        _fail("MISSING_FULL_CASE",
              "unit %r full cases do not cover the query states (missing=%r extra=%r)"
              % (unit_id, missing, extra))

    histories = doc.get("histories")
    if not isinstance(histories, Mapping):
        _fail("MISSING_HISTORIES", "doc['histories'] is absent")

    actions = {action_id: {"cost": cost_tokens[action_id], "outcomes": {}}
               for action_id in action_order}
    truth = {}
    for state in states:
        case = cases[state]
        history_id = case.get("history_id")
        if not isinstance(history_id, str) or history_id not in histories:
            _fail("MISSING_HISTORY", "unit %r state %r history_id %r is absent"
                  % (unit_id, state, history_id))
        history = histories[history_id]
        if history.get("endpoint") != "full":
            _fail("HISTORY_ENDPOINT_NOT_FULL",
                  "unit %r state %r history %r endpoint is %r, expected 'full'"
                  % (unit_id, state, history_id, history.get("endpoint")))
        window_ids = history.get("window_ids")
        if not isinstance(window_ids, Sequence) or isinstance(window_ids, (str, bytes)):
            _fail("WINDOW_SET_MISMATCH", "unit %r state %r has no window_ids"
                  % (unit_id, state))
        window_ids = list(window_ids)
        if sorted(window_ids) != action_order:
            _fail("WINDOW_SET_MISMATCH",
                  "unit %r state %r windows %r differ from the declared set %r"
                  % (unit_id, state, window_ids, action_order))
        if window_ids != action_order:
            _fail("WINDOW_ORDER_MISMATCH",
                  "unit %r state %r windows are out of declared order: %r"
                  % (unit_id, state, window_ids))
        case_windows = case.get("history_window_ids")
        if case_windows is not None and list(case_windows) != window_ids:
            _fail("WINDOW_SET_MISMATCH",
                  "unit %r state %r case windows disagree with its history"
                  % (unit_id, state))
        signatures = history.get("signatures")
        if not isinstance(signatures, Sequence) or isinstance(signatures, (str, bytes)):
            _fail("SIGNATURE_COUNT_MISMATCH",
                  "unit %r state %r has no signature table" % (unit_id, state))
        if len(signatures) != len(window_ids):
            _fail("SIGNATURE_COUNT_MISMATCH",
                  "unit %r state %r has %d signatures for %d windows"
                  % (unit_id, state, len(signatures), len(window_ids)))
        for index, action_id in enumerate(window_ids):
            actions[action_id]["outcomes"][state] = signatures[index]
        truth[state] = case.get("truth_3state")

    return _assemble(ADAPTER_W38, doc.get("family_id"), unit_id, states, labels, priors,
                     actions, source, truth)


def adapt_real_trace_manifest(doc, unit_id: str, *, source=None) -> dict:
    """Convert one REAL_TRACE real_trace_signature_by_window unit."""
    units = doc.get("units")
    if not isinstance(units, Mapping) or unit_id not in units:
        _fail("UNIT_NOT_FOUND", "unit %r is absent from doc['units']" % (unit_id,))
    unit = units[unit_id]
    states, labels, priors = _unit_query(unit, unit_id)

    action_order, cost_tokens = _declared_costs(doc.get("window_costs"),
                                                "doc['window_costs']")
    positions = doc.get("positions")
    if isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
        if set(positions) != set(states):
            _fail("POSITIONS_MISMATCH",
                  "unit %r query states %r differ from doc positions %r"
                  % (unit_id, states, list(positions)))

    windows = unit.get("windows")
    if isinstance(windows, Mapping):
        if sorted(windows) != action_order:
            _fail("WINDOW_SET_MISMATCH",
                  "unit %r window set %r differs from doc window_costs %r"
                  % (unit_id, sorted(windows), action_order))
        for action_id in action_order:
            declared = windows[action_id]
            if isinstance(declared, Mapping) and "cost" in declared:
                if declared["cost"] != cost_tokens[action_id]:
                    _fail("WINDOW_COST_MISMATCH",
                          "unit %r window %r cost %r disagrees with doc window_costs %r"
                          % (unit_id, action_id, declared["cost"],
                             cost_tokens[action_id]))

    cases = _full_case_index(doc, unit_id, ADAPTER_REAL)
    if set(cases) != set(states):
        missing = sorted(set(states) - set(cases))
        extra = sorted(set(cases) - set(states))
        _fail("MISSING_OUTCOME",
              "unit %r cases do not cover the query states (missing=%r extra=%r)"
              % (unit_id, missing, extra))

    actions = {action_id: {"cost": cost_tokens[action_id], "outcomes": {}}
               for action_id in action_order}
    truth = {}
    for state in states:
        case = cases[state]
        signature = case.get("signature_by_window")
        if not isinstance(signature, Mapping):
            _fail("WINDOW_SET_MISMATCH",
                  "unit %r state %r has no signature_by_window" % (unit_id, state))
        if sorted(signature) != action_order:
            _fail("WINDOW_SET_MISMATCH",
                  "unit %r state %r signature windows %r differ from the declared set"
                  % (unit_id, state, sorted(signature)))
        truth_value = case.get("truth_Q")
        if truth_value is None or int(truth_value) != labels[state]:
            _fail("TRUTH_QUERY_MISMATCH",
                  "unit %r state %r truth_Q %r disagrees with query label %r"
                  % (unit_id, state, truth_value, labels[state]))
        for action_id in action_order:
            actions[action_id]["outcomes"][state] = signature[action_id]
        truth[state] = case.get("truth_3state")

    return _assemble(ADAPTER_REAL, doc.get("family_id"), unit_id, states, labels, priors,
                     actions, source, truth)


_ADAPTER_FUNCS = {
    ADAPTER_W38: adapt_w38_manifest,
    ADAPTER_REAL: adapt_real_trace_manifest,
}


def adapt_unit(doc, kind: str, unit_id: str, *, source=None) -> dict:
    """Dispatch one unit to the adapter named by kind."""
    if kind not in _ADAPTER_FUNCS:
        _fail("UNKNOWN_KIND", "unknown adapter kind %r (expected one of %r)"
              % (kind, list(ADAPTERS)))
    return _ADAPTER_FUNCS[kind](doc, unit_id, source=source)


def load_manifest(path: Any, kind: str) -> dict:
    """Read a frozen manifest as JSON."""
    if kind not in ADAPTERS:
        _fail("UNKNOWN_KIND", "unknown adapter kind %r (expected one of %r)"
              % (kind, list(ADAPTERS)))
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        _fail("MANIFEST_UNREADABLE", "cannot read manifest %s: %s" % (path, exc))
    try:
        doc = json.loads(text)
    except ValueError as exc:
        _fail("MANIFEST_JSON_INVALID", "manifest %s is not valid JSON: %s" % (path, exc))
    if not isinstance(doc, dict):
        _fail("MANIFEST_JSON_INVALID", "manifest %s is not a JSON object" % (path,))
    return doc


def adapt_manifest(path: Any, kind: str, unit_id: str) -> dict:
    """Read path, adapt unit_id and stamp the source block."""
    doc = load_manifest(path, kind)
    source = {
        "manifest": _repo_relative(path),
        "sha256": sha256_file(path),
        "family_id": doc.get("family_id"),
        "unit_id": unit_id,
        "adapter": kind,
    }
    return adapt_unit(doc, kind, unit_id, source=source)


# --------------------------------------------------------------------------- #
# Unified-document validation and loading
# --------------------------------------------------------------------------- #

def _pairs_hook(pairs):
    """object_pairs_hook that refuses duplicate keys anywhere in the JSON."""
    seen = {}
    duplicates = []
    for key, value in pairs:
        if key in seen:
            duplicates.append(key)
        seen[key] = value
    if duplicates:
        raise SchemaError(
            "DUPLICATE_OBSERVED_ACTION",
            "unified JSON repeats object key(s) %r; observed/actions must not be "
            "silently de-duplicated" % (sorted(set(duplicates)),))
    return seen


def _validate_boundary(boundary) -> None:
    if not isinstance(boundary, Mapping):
        _fail("MISSING_FIELD", "boundary must be an object")
    for field in ("domain", "is_field_evidence", "physical_quantities_allowed"):
        if field not in boundary:
            _fail("MISSING_FIELD", "boundary is missing %r" % field)
    if boundary["domain"] not in BOUNDARY_DOMAINS:
        _fail("BOUNDARY_VIOLATION", "boundary domain %r is not declared"
              % (boundary["domain"],))
    if boundary["is_field_evidence"] is not False:
        _fail("BOUNDARY_VIOLATION", "boundary is_field_evidence must be false")
    if boundary["physical_quantities_allowed"] is not False:
        _fail("BOUNDARY_VIOLATION", "boundary physical_quantities_allowed must be false")


def _validate_unified(doc) -> None:
    """Validate a unified world document against the frozen schema."""
    if not isinstance(doc, Mapping):
        _fail("SCHEMA_MISMATCH", "unified document must be a JSON object")
    for field in _MANIFEST_FIELDS:
        if field not in doc:
            _fail("MISSING_FIELD", "unified document is missing %r" % field)
    if doc["schema"] != SCHEMA_ID:
        _fail("SCHEMA_MISMATCH", "schema %r is not %r" % (doc["schema"], SCHEMA_ID))

    states = doc["states"]
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        _fail("SCHEMA_MISMATCH", "states must be an array")
    states = list(states)
    if not states:
        _fail("EMPTY_STATES", "states must be non-empty")
    if not all(isinstance(state, str) and state for state in states):
        _fail("INVALID_STATE_ID", "every state id must be a non-empty string")
    if len(set(states)) != len(states):
        _fail("DUPLICATE_STATE", "states contains duplicate ids: %r" % (states,))

    labels = doc["labels"]
    if not isinstance(labels, Mapping):
        _fail("INVALID_LABEL", "labels must be an object")
    if set(labels) != set(states):
        _fail("INVALID_LABEL", "labels keys %r do not match states %r"
              % (sorted(labels), sorted(states)))
    for state in states:
        value = labels[state]
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            _fail("INVALID_LABEL", "label for state %r is %r, expected 0/1"
                  % (state, value))

    priors = doc["priors"]
    if not isinstance(priors, Mapping):
        _fail("INVALID_PRIOR", "priors must be an object")
    if set(priors) != set(states):
        _fail("INVALID_PRIOR", "priors keys do not match states")
    total = Fraction(0)
    for state in states:
        parsed, _token = _fraction_token(priors[state], code="INVALID_PRIOR",
                                         nonpositive_code="INVALID_PRIOR",
                                         where="priors[%r]" % (state,))
        total += parsed
    if total != 1:
        _fail("PRIOR_NOT_NORMALISED", "priors sum to %s, expected 1" % (total,))

    actions = doc["actions"]
    if not isinstance(actions, Mapping) or not actions:
        _fail("EMPTY_ACTIONS", "actions must be a non-empty object")
    for action_id, spec in actions.items():
        if not isinstance(action_id, str) or not action_id:
            _fail("INVALID_ACTION_ID", "action id must be a non-empty string")
        if not isinstance(spec, Mapping):
            _fail("SCHEMA_MISMATCH", "action %r must be an object" % (action_id,))
        if "cost" not in spec:
            _fail("MISSING_COST", "action %r has no cost" % (action_id,))
        _fraction_token(spec["cost"], code="INVALID_COST",
                        nonpositive_code="NONPOSITIVE_COST",
                        where="actions[%r].cost" % (action_id,))
        outcomes = spec.get("outcomes")
        if not isinstance(outcomes, Mapping):
            _fail("MISSING_OUTCOME", "action %r has no outcomes object" % (action_id,))
        if set(outcomes) != set(states):
            missing = sorted(set(states) - set(outcomes))
            unknown = sorted(set(outcomes) - set(states))
            _fail("MISSING_OUTCOME",
                  "action %r outcomes do not cover every state (missing=%r unknown=%r)"
                  % (action_id, missing, unknown))

    observed = doc["observed"]
    if not isinstance(observed, Mapping):
        _fail("SCHEMA_MISMATCH", "observed must be an object")
    for action_id, outcome in observed.items():
        if action_id not in actions:
            _fail("UNKNOWN_ACTION", "observed action %r is absent from the action table"
                  % (action_id,))
        outcomes = actions[action_id]["outcomes"]
        if not any(outcomes[state] == outcome for state in states):
            _fail("OBSERVATION_CONFLICT",
                  "observed outcome %r for action %r matches no candidate world"
                  % (outcome, action_id))

    _validate_boundary(doc["boundary"])

    source = doc["source"]
    if not isinstance(source, Mapping):
        _fail("SCHEMA_MISMATCH", "source must be an object")
    if source.get("adapter") not in ADAPTERS:
        _fail("UNKNOWN_KIND", "source.adapter %r is not a declared adapter"
              % (source.get("adapter"),))
    for field in ("family_id", "unit_id"):
        if not isinstance(source.get(field), str) or not source[field]:
            _fail("SCHEMA_MISMATCH", "source.%s must be a non-empty string" % field)


def load_unified(path: Any) -> dict:
    """Load and validate a unified world JSON file (duplicate keys are fatal)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        _fail("UNIFIED_UNREADABLE", "cannot read unified document %s: %s" % (path, exc))
    try:
        doc = json.loads(text, object_pairs_hook=_pairs_hook)
    except SchemaError:
        raise
    except ValueError as exc:
        _fail("UNIFIED_JSON_INVALID", "unified document %s is not valid JSON: %s"
              % (path, exc))
    _validate_unified(doc)
    return doc


# --------------------------------------------------------------------------- #
# Manifest kernel partition / three-state
# --------------------------------------------------------------------------- #

def _core_actions(unified, states) -> dict:
    """Strip the unified actions down to the shape the core module expects."""
    actions = {}
    for action_id, spec in unified["actions"].items():
        actions[action_id] = {
            "cost": spec["cost"],
            "outcomes": {state: spec["outcomes"][state] for state in states},
        }
    return actions


def full_classes(unified) -> dict:
    """Return {state: [states with the identical full declared signature]}.

    Calls the core equivalent_set with every declared action observed for one
    state; that filter is the manifest equivalence kernel cell.
    """
    _validate_unified(unified)
    states = list(unified["states"])
    actions = _core_actions(unified, states)
    action_ids = list(actions)
    classes = {}
    for state in states:
        observed = {action_id: actions[action_id]["outcomes"][state]
                    for action_id in action_ids}
        classes[state] = list(equivalent_set(states, actions, observed))
    return classes


def compute_three_state(unified) -> dict:
    """Return the adapter-contract three-state of every state.

    DEC_CONNECTED / DEC_DISCONNECTED when the full-signature class is
    label-pure, else UNDECIDABLE_BY_DECLARED_ACTIONS.
    """
    labels = {state: int(unified["labels"][state]) for state in unified["states"]}
    three_state = {}
    for state, members in full_classes(unified).items():
        member_labels = {labels[member] for member in members}
        if member_labels == {1}:
            three_state[state] = DEC_CONNECTED
        elif member_labels == {0}:
            three_state[state] = DEC_DISCONNECTED
        else:
            three_state[state] = UNDECIDABLE_BY_DECLARED_ACTIONS
    return three_state
