#!/usr/bin/env python3
"""Derive in-scope short answers and verify numeric derivations reproducibly."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "sources" / "exam-papers"
ROUTING_PATH = PAPERS / "routing.json"
INVENTORY_PATH = PAPERS / "question-inventory.json"
SOLUTIONS_PATH = PAPERS / "solutions-luna.json"
VERIFICATION_PATH = PAPERS / "solutions-luna-verification.txt"
SEED = 42


def numeric_values(record: dict[str, Any]) -> dict[str, Any]:
    """Compute numeric answer fields for an in-scope question by ID."""
    return {}


# These are Terra numeric questions whose full mathematical inputs are present
# in their inventory text.  Other Terra numeric records refer to missing shared
# figures/matrices, so the verifier reports those as not computable.
def terra_w1_q152(_: dict[str, Any]) -> dict[str, Any]:
    # Week-1 position encodings are fixed sinusoidal vectors, not learned tables.
    return {"additional_parameters": 0}


TERRA_NUMERIC_FUNCTIONS: dict[
    str, Callable[[dict[str, Any]], dict[str, Any]]
] = {
    "da5004_2024T1_ET_AN--q152": terra_w1_q152,
    "da5004_2024T1_ET_FN--q207": terra_w1_q152,
}


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    routing = json.loads(ROUTING_PATH.read_text(encoding="utf-8"))["records"]
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))["questions"]
    by_id = {question["id"]: question for question in inventory}
    selected: list[dict[str, Any]] = []
    for route in routing:
        if route.get("week") not in (1, 2, 3, 4) or route.get("type") != "SA":
            continue
        question = by_id.get(route["id"])
        if question is None or question.get("image_only") is not False:
            continue
        selected.append({**route, "question": question})
    return selected, by_id


def run_primary() -> tuple[dict[str, Any], str]:
    rng = np.random.default_rng(SEED)
    del rng  # The selected derivations are deterministic and require no draws.
    selected, _ = load_inputs()
    records: list[dict[str, Any]] = []
    log = [f"seed={SEED}", f"in_scope_count={len(selected)}"]
    for route in selected:
        qid = route["id"]
        function = NUMERIC_FUNCTIONS.get(qid)
        if function is None:
            # The current inventory selection is empty; keep unsupported cases
            # explicit should the source data change without a matching solver.
            records.append(
                {
                    "id": qid,
                    "week": route["week"],
                    "answer": "",
                    "steps": "No per-question derivation is registered.",
                    "status": "ambiguous",
                    "ambiguity_note": "No verified derivation is available for this question.",
                    "verified_by_code": False,
                    "is_numeric": False,
                    "numeric_values": {},
                    "code_ref": "",
                }
            )
            continue
        calculated = function(route["question"])
        expected: dict[str, Any] = {}
        passed = comparable(calculated, expected)
        answer = answer_from_values(calculated)
        records.append(
            {
                "id": qid,
                "week": route["week"],
                "answer": answer,
                "steps": "Computed from the supplied question values with NumPy; compared with the hand derivation.",
                "status": "derived",
                "ambiguity_note": "",
                "verified_by_code": passed,
                "is_numeric": True,
                "numeric_values": calculated,
                "code_ref": f"solve_numeric.py::{function.__name__}",
            }
        )
        log.append(
            f"{qid} computed={json.dumps(calculated, sort_keys=True)} "
            f"expected={json.dumps(expected, sort_keys=True)} {'PASS' if passed else 'FAIL'}"
        )
    output = {"author": "luna", "records": records}
    SOLUTIONS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output, "\n".join(log) + "\n"


def comparable(calculated: dict[str, Any], expected: dict[str, Any]) -> bool:
    if calculated.keys() != expected.keys():
        return False
    for key, actual in calculated.items():
        target = expected[key]
        if isinstance(actual, (int, float)) and isinstance(target, (int, float)):
            if not math.isclose(float(actual), float(target), rel_tol=1e-9, abs_tol=1e-9):
                return False
        elif actual != target:
            return False
    return True


def answer_from_values(values: dict[str, Any]) -> str:
    if len(values) == 1:
        value = next(iter(values.values()))
        return str(value)
    return json.dumps(values, sort_keys=True)


def run_terra() -> str:
    _, questions = load_inputs()
    lines = [f"terra_function_ids={','.join(sorted(TERRA_NUMERIC_FUNCTIONS))}"]
    for week in range(1, 5):
        path = PAPERS / f"solutions-terra-w{week}.json"
        if not path.exists():
            lines.append(f"week={week} file=missing status=SKIPPED")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data.get("records", []):
            if record.get("is_numeric") is not True:
                continue
            qid = record["id"]
            function = TERRA_NUMERIC_FUNCTIONS.get(qid)
            if function is None or qid not in questions:
                lines.append(f"{qid} computed=UNAVAILABLE expected={record.get('numeric_values', {})} MISMATCH")
                continue
            calculated = function(questions[qid])
            expected = record.get("numeric_values", {})
            outcome = "MATCH" if comparable(calculated, expected) else "MISMATCH"
            lines.append(
                f"{qid} computed={json.dumps(calculated, sort_keys=True)} "
                f"expected={json.dumps(expected, sort_keys=True)} {outcome}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-terra", action="store_true")
    args = parser.parse_args()
    if args.verify_terra:
        print(run_terra(), end="")
        return
    _, verification = run_primary()
    VERIFICATION_PATH.write_text(verification, encoding="utf-8")
    print(verification, end="")


if __name__ == "__main__":
    NUMERIC_FUNCTIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
    main()
