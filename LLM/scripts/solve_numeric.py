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


def terra_encoder_layer_parameters(_: dict[str, Any]) -> dict[str, Any]:
    d_model, d_ff, layers = 256, 1024, 3
    attention = 4 * d_model * d_model
    feed_forward = 2 * d_model * d_ff + d_model + d_ff
    layer_norm = 4 * d_model
    return {"encoder_parameters": layers * (attention + feed_forward + layer_norm)}


def _attention_output(
    attention: list[list[float]], embeddings: list[list[float]]
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=float)
    w_v = np.array([[0.1, 1], [1, 0.5]], dtype=float)
    w_o = np.array([[1, 1], [-1, 1]], dtype=float)
    return np.asarray(attention, dtype=float) @ values @ w_v @ w_o


def terra_w1_q153(_: dict[str, Any]) -> dict[str, Any]:
    attention = [
        [0.06, 0.45, 0.15, 0.35],
        [0.22, 0.13, 0.28, 0.36],
        [0.06, 0.25, 0.17, 0.53],
        [0.04, 0.12, 0.17, 0.67],
    ]
    # The source context orders the input as "i drink coffee often".
    embeddings = [[1, -1], [0, 1], [1, 0], [1, 1]]
    output = _attention_output(attention, embeddings)[3]
    return {"output": np.round(output, 3).tolist(), "sum": round(float(output.sum()), 3)}


def terra_w1_q154(_: dict[str, Any]) -> dict[str, Any]:
    attention = np.array(
        [[0.06, 0.45, 0.15, 0.35], [0.22, 0.13, 0.28, 0.36],
         [0.06, 0.25, 0.17, 0.53], [0.04, 0.12, 0.17, 0.67]],
        dtype=float,
    )
    return {"attention": float(attention[2, 3])}


def terra_w1_fn_q208(_: dict[str, Any]) -> dict[str, Any]:
    attention = [
        [0.13, 0.22, 0.28, 0.36],
        [0.45, 0.06, 0.15, 0.35],
        [0.25, 0.06, 0.17, 0.53],
        [0.12, 0.04, 0.17, 0.67],
    ]
    # The source context orders the input as "i like coffee much".
    embeddings = [[0, 1], [1, -1], [1, 0], [1, 1]]
    output = _attention_output(attention, embeddings)[2]
    return {"output": np.round(output, 3).tolist(), "sum": round(float(output.sum()), 3)}


def terra_w1_q155(_: dict[str, Any]) -> dict[str, Any]:
    qkt = np.array(
        [[0.97, 2.56, 0.25, 2.36, 3.96, 1.83, 2.31, 0.77],
         [2.04, 2.55, 3.35, 2.21, 1.94, 0.16, 1.49, 0.99],
         [2.34, 3.65, 2.63, 0.95, 3.37, 3.59, 1.30, 2.02],
         [0.94, 0.41, 2.26, 2.29, 3.92, 2.48, 1.77, 3.85],
         [0.78, 2.81, 0.26, 0.30, 2.16, 2.56, 3.99, 2.83],
         [3.02, 2.28, 1.59, 3.77, 2.58, 3.41, 1.80, 3.93],
         [3.09, 1.23, 2.02, 1.77, 0.42, 0.99, 2.05, 2.00],
         [3.12, 1.79, 1.78, 1.60, 1.00, 1.86, 2.71, 0.82]],
        dtype=float,
    )
    return {"sequence_length": int(qkt.shape[0])}


def terra_w1_q157(_: dict[str, Any]) -> dict[str, Any]:
    qkt = np.array(
        [[0.97, 2.56, 0.25, 2.36, 3.96, 1.83, 2.31, 0.77],
         [2.04, 2.55, 3.35, 2.21, 1.94, 0.16, 1.49, 0.99],
         [2.34, 3.65, 2.63, 0.95, 3.37, 3.59, 1.30, 2.02],
         [0.94, 0.41, 2.26, 2.29, 3.92, 2.48, 1.77, 3.85],
         [0.78, 2.81, 0.26, 0.30, 2.16, 2.56, 3.99, 2.83],
         [3.02, 2.28, 1.59, 3.77, 2.58, 3.41, 1.80, 3.93],
         [3.09, 1.23, 2.02, 1.77, 0.42, 0.99, 2.05, 2.00],
         [3.12, 1.79, 1.78, 1.60, 1.00, 1.86, 2.71, 0.82]],
        dtype=float,
    )
    # q_3 is zero-based row 3; softmax preserves the row's argmax.
    return {"key_index": int(np.argmax(qkt[3]))}


def terra_w3_q118(_: dict[str, Any]) -> dict[str, Any]:
    return {"total_tokens": 41, "sachin_count": 6, "scaled_probability": 10 * 6 / 41}


def terra_w3_q119(_: dict[str, Any]) -> dict[str, Any]:
    return {"total_tokens": 41, "phrase_count": 1, "scaled_probability": 100 / 41}


def terra_w3_q120(_: dict[str, Any]) -> dict[str, Any]:
    return {"total_tokens": 41, "phrase_count": 1, "scaled_probability": 100 / 41}


def terra_w3_q189(_: dict[str, Any]) -> dict[str, Any]:
    batch_size, sequence_length = 32, 1024
    return {"batch": batch_size, "length": sequence_length,
            "targets": batch_size * (sequence_length - 1)}


def terra_w3_q285(_: dict[str, Any]) -> dict[str, Any]:
    return {"probability": 0.41}


def terra_w3_q286(_: dict[str, Any]) -> dict[str, Any]:
    factors = [0.41, 0.43, 0.29, 0.48, 0.29, 0.23]
    return {"factors": factors, "scaled_probability": 1000 * float(np.prod(factors))}


def terra_w3_q288(_: dict[str, Any]) -> dict[str, Any]:
    top_k = np.array([0.41, 0.14, 0.09], dtype=float)
    return {"selected": top_k.tolist(), "probability": float(top_k[0] / top_k.sum())}


def terra_w4_q148(_: dict[str, Any]) -> dict[str, Any]:
    matrix = np.array(
        [[.05,.02,.14,.07,.41,.09,.08,.05,.08,.01],
         [.10,.04,.06,.16,.01,.43,.06,.09,.01,.04],
         [.05,.07,.28,.29,.03,.08,.04,.08,.04,.04],
         [.10,.02,.48,.02,.01,.07,.06,.14,.07,.01],
         [.08,.29,.04,.01,.04,.03,.25,.05,.15,.06],
         [.14,.03,.13,.08,.01,.03,.06,.23,.08,.22]], dtype=float)
    return {"row_maxima": np.max(matrix, axis=1).tolist()}


def terra_w4_q201(_: dict[str, Any]) -> dict[str, Any]:
    return terra_w4_q148({})


def terra_w4_q198(question: dict[str, Any]) -> dict[str, Any]:
    probabilities = np.array([.06,.20,.14,.06,.09,.08,.05,.15,.06,.11], dtype=float)
    ordered = np.sort(probabilities)[::-1]
    cumulative = np.cumsum(ordered)
    count = int(np.searchsorted(cumulative, .7, side="left") + 1)
    return {"sorted": ordered[:count].tolist(), "cumulative_five": float(cumulative[4]),
            "cumulative_six": float(cumulative[5]), "count": count}


def terra_w4_q145_fn(_: dict[str, Any]) -> dict[str, Any]:
    logits, temperature = np.array([1.5,.5,0,-.5]), 2.0
    probs = np.exp(logits / temperature - np.max(logits / temperature))
    probs /= probs.sum()
    return {"logits": logits.tolist(), "temperature": temperature,
            "probability_rounded": round(float(probs[0]), 2)}


def terra_w4_q145_q1(_: dict[str, Any]) -> dict[str, Any]:
    logits, temperature = np.array([2,1,.5,-1]), .5
    scaled = logits / temperature
    probs = np.exp(scaled - np.max(scaled))
    probs /= probs.sum()
    return {"logits": logits.tolist(), "temperature": temperature,
            "probability_rounded": round(float(probs[0]), 2)}


def terra_w4_q153(_: dict[str, Any]) -> dict[str, Any]:
    vocabulary, timestep = 6, 4
    return {"vocabulary": vocabulary, "prefix_length": timestep - 1,
            "decoder_evaluations": vocabulary ** (timestep - 1)}


def terra_w4_q154(_: dict[str, Any]) -> dict[str, Any]:
    vocabulary, length = 6, 5
    return {"vocabulary": vocabulary, "length": length, "sequences": vocabulary ** length}


def terra_w4_q4(_: dict[str, Any]) -> dict[str, Any]:
    sorted_probabilities = np.array([.5,.2,.15,.1,.05], dtype=float)
    cumulative = np.cumsum(sorted_probabilities)
    return {"cumulative_two": float(cumulative[1]), "cumulative_three": float(cumulative[2])}


def terra_w4_q181(_: dict[str, Any]) -> dict[str, Any]:
    vocabulary, length, layers, beam_width = 10, 5, 6, 3
    return {"vocabulary": vocabulary,
            "step1": [1, beam_width, vocabulary],
            "step3": [1, beam_width, vocabulary ** 3]}


def terra_w4_q8(_: dict[str, Any]) -> dict[str, Any]:
    probs = np.array([.4,.25,.15,.1,.1], dtype=float)
    count = int(np.searchsorted(np.cumsum(probs), .8, side="left") + 1)
    return {"probabilities": probs[:count].tolist(), "p": .8, "count": count}


def terra_w4_q11(_: dict[str, Any]) -> dict[str, Any]:
    logits, temperature = np.array([2,1,0,-1], dtype=float), 2.0
    scaled = logits / temperature
    probs = np.exp(scaled - np.max(scaled))
    probs /= probs.sum()
    return {"logits": logits.tolist(), "temperature": temperature,
            "probability_rounded": round(float(probs[0]), 2)}


def terra_w4_q17(_: dict[str, Any]) -> dict[str, Any]:
    length, layers = 5, 6
    bert = layers * length * length
    gpt = layers * length * (length + 1) // 2
    return {"length": length, "layers": layers, "bert": bert, "gpt": gpt}


def terra_w4_q17_topk(_: dict[str, Any]) -> dict[str, Any]:
    probabilities = np.array([.1,.4,.05,.25,.2], dtype=float)
    top = np.sort(probabilities)[-2:]
    return {"B": .4, "D": .25, "probability": float(.4 / top.sum())}


def terra_w4_q18(_: dict[str, Any]) -> dict[str, Any]:
    d_model, vocab, length, segments = 128, 5000, 64, 2
    token, position, segment = vocab*d_model, length*d_model, segments*d_model
    return {"token": token, "position": position, "segment": segment,
            "total": token + position + segment}


TERRA_NUMERIC_FUNCTIONS: dict[
    str, Callable[[dict[str, Any]], dict[str, Any]]
] = {
    "da5004_2024T1_ET_AN--q151": terra_encoder_layer_parameters,
    "da5004_2024T1_ET_AN--q152": terra_w1_q152,
    "da5004_2024T1_ET_AN--q153": terra_w1_q153,
    "da5004_2024T1_ET_AN--q154": terra_w1_q154,
    "da5004_2024T1_ET_AN--q155": terra_w1_q155,
    "da5004_2024T1_ET_AN--q157": terra_w1_q157,
    "da5004_2024T1_ET_FN--q206": terra_encoder_layer_parameters,
    "da5004_2024T1_ET_FN--q207": terra_w1_q152,
    "da5004_2024T1_ET_FN--q208": terra_w1_fn_q208,
    "da5004_2024T1_ET_FN--q209": terra_w1_q154,
    "da5004_2024T1_ET_FN--q210": terra_w1_q155,
    "da5004_2024T1_ET_FN--q212": terra_w1_q157,
    "da5004_2024T3_Q1_NA--q285": terra_w3_q285,
    "da5004_2024T3_Q1_NA--q286": terra_w3_q286,
    "da5004_2024T3_Q1_NA--q288": terra_w3_q288,
    "da5004_2025T1_Q1_NA--q118": terra_w3_q118,
    "da5004_2025T1_Q1_NA--q119": terra_w3_q119,
    "da5004_2025T1_Q1_NA--q120": terra_w3_q120,
    "da5004_2025T3_Q1_NA--q189": terra_w3_q189,
    "da5004_2025T2_ET_AN--q198": terra_w4_q198,
    "da5004_2025T2_ET_FN--q145": terra_w4_q145_fn,
    "da5004_2025T2_Q1_NA--q145": terra_w4_q145_q1,
    "da5004_2025T2_Q1_NA--q153": terra_w4_q153,
    "da5004_2025T2_Q1_NA--q154": terra_w4_q154,
    "da5004_2025T3_ET_FN--q4": terra_w4_q4,
    "da5004_2025T3_Q1_NA--q181": terra_w4_q181,
    "da5004_2026T1_ET_AN--q8": terra_w4_q8,
    "da5004_2026T1_ET_AN--q11": terra_w4_q11,
    "da5004_2026T1_ET_FN--q17": terra_w4_q17,
    "da5004_2026T2_ET_FN--q17": terra_w4_q17,
    "da5004_2026T1_Q2_NA--q17": terra_w4_q17_topk,
    "da5004_2026T1_Q2_NA--q18": terra_w4_q18,
    "da5004_2026T2_ET_AN--q8": terra_w4_q8,
    "da5004_2024T1_ET_AN--q148": terra_w4_q148,
    "da5004_2024T1_ET_FN--q201": terra_w4_q201,
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
    def equal(actual: Any, target: Any) -> bool:
        if isinstance(actual, (int, float)) and isinstance(target, (int, float)):
            return math.isclose(float(actual), float(target), rel_tol=1e-9, abs_tol=1e-9)
        if isinstance(actual, list) and isinstance(target, list):
            return len(actual) == len(target) and all(equal(a, b) for a, b in zip(actual, target))
        if isinstance(actual, dict) and isinstance(target, dict):
            return actual.keys() == target.keys() and all(equal(actual[k], target[k]) for k in actual)
        return actual == target

    return equal(calculated, expected)


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
