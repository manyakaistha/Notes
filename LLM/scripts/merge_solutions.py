#!/usr/bin/env python3
"""Merge routed Weeks 1–4 answer records and emit an audit report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from solve_numeric import PAPERS, ROUTING_PATH, INVENTORY_PATH, run_terra, run_overrides

OUTPUT = PAPERS / "question-solutions.json"
REPORT = PAPERS / "solutions-report.txt"
LUNA_VERIFICATION = PAPERS / "solutions-luna-verification.txt"
OVERRIDES = PAPERS / "solutions-overrides.json"
SOURCES = [
    "solutions-terra-w1-a.json", "solutions-terra-w1-b.json",
    "solutions-terra-w2-a.json", "solutions-terra-w2-b.json",
    "solutions-terra-w3.json", "solutions-terra-w4.json",
]
FIELDS = ["id", "week", "answer", "steps", "status", "ambiguity_note",
          "verified_by_code", "verification_note", "answer_options", "is_numeric",
          "numeric_values", "author", "code_ref"]


def main() -> None:
    routes = [r for r in json.loads(ROUTING_PATH.read_text(encoding="utf-8"))["records"]
              if r.get("week") in (1, 2, 3, 4)]
    expected = {r["id"]: r for r in routes}
    questions = {q["id"]: q for q in json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))["questions"]}
    source_records: dict[str, tuple[dict[str, Any], str, str]] = {}
    duplicates: list[str] = []
    for filename in SOURCES:
        payload = json.loads((PAPERS / filename).read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            qid = record["id"]
            if qid in source_records:
                duplicates.append(qid)
            source_records[qid] = (record, str(payload.get("author", "terra")), filename)

    missing = sorted(set(expected) - set(source_records))
    extra = sorted(set(source_records) - set(expected))
    assert len(routes) == 241, f"routing scope should be 241, got {len(routes)}"
    assert not duplicates, f"duplicate source records: {duplicates}"
    assert not missing, f"missing solution records: {missing}"
    assert not extra, f"extra solution records: {extra}"

    override_data = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    overrides = {r["id"]: r for r in override_data.get("overrides", [])}
    verification_output = run_terra()
    override_verification_output = run_overrides()
    LUNA_VERIFICATION.write_text(verification_output + "\nOverride-controlled numeric derivations:\n" + override_verification_output, encoding="utf-8")
    override_checked: dict[str, str] = {}
    for line in override_verification_output.splitlines():
        if " MATCH" in line:
            override_checked[line.split(" computed=", 1)[0]] = "MATCH"
        elif " MISMATCH" in line:
            override_checked[line.split(" computed=", 1)[0]] = "MISMATCH"
        elif " UNVERIFIED " in line:
            override_checked[line.split(" UNVERIFIED ", 1)[0]] = "UNVERIFIED"
    checked: dict[str, str] = {}
    for line in verification_output.splitlines():
        if " computed=" not in line:
            continue
        qid, _, outcome = line.partition(" computed=")
        checked[qid] = "MATCH" if outcome.endswith(" MATCH") else "MISMATCH"

    records: list[dict[str, Any]] = []
    for qid, route in expected.items():
        original, author, filename = source_records[qid]
        original = dict(original)
        override = overrides.get(qid)
        if override:
            for field in ("answer", "steps", "status", "ambiguity_note", "answer_options", "is_numeric", "numeric_values"):
                if field in override:
                    original[field] = override[field]
        status = original.get("status", "")
        assert status in {"derived", "ambiguous"}, f"unsupported status {status}: {qid}"
        numeric = bool(original.get("is_numeric", False))
        answer_options = list(original.get("answer_options") or [])
        note = ""
        verified = (override_checked.get(qid) == "MATCH" if qid in overrides and original.get("is_numeric")
                    else checked.get(qid) == "MATCH")
        if verified:
            note = "solve_numeric.py recomputed adjudicated inputs/conventions; all recorded values and adjudication checks MATCH."
        elif numeric and override_checked.get(qid) == "UNVERIFIED":
            note = "Override numeric derivation not code-verified: complete numeric inputs are absent from the editable extraction; manual scan adjudication retained."
        elif numeric:
            q = questions.get(qid, {})
            if q.get("stem_images") or q.get("image_only"):
                note = "depends on figure, no code path"
            else:
                note = "solve_numeric.py has no registered derivation for this ID; the extracted question data do not supply a complete computable input set."
        elif status == "ambiguous":
            note = "No numeric checker path; answer remains explicitly ambiguous."
        else:
            note = "Non-numeric response; no numeric checker path applies."

        item = {
            "id": qid,
            "week": route["week"],
            "answer": original.get("answer", ""),
            "steps": original.get("steps", ""),
            "status": status,
            "ambiguity_note": original.get("ambiguity_note", ""),
            "verified_by_code": verified,
            "verification_note": note,
            "answer_options": answer_options,
            "is_numeric": numeric,
            "numeric_values": original.get("numeric_values", {}),
            "author": author,
            "code_ref": original.get("code_ref", f"{filename}::{qid}"),
        }
        if override:
            item["override_source"] = override.get("source", "solutions-overrides.json")
        if route.get("type") in {"MCQ", "MSQ"} and not answer_options:
            assert status == "ambiguous" and item["ambiguity_note"].strip(), f"empty options on non-ambiguous MCQ/MSQ: {qid}"
        records.append(item)

    assert len(records) == 241
    assert len({r["id"] for r in records}) == 241
    assert {r["id"] for r in records} == set(expected)
    assert all(list(r) == FIELDS + (["override_source"] if r["id"] in overrides else []) for r in records)
    OUTPUT.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    week_status = Counter((r["week"], r["status"]) for r in records)
    type_status = Counter((expected[r["id"]].get("type", "unknown"), r["status"]) for r in records)
    numeric_count = sum(r["is_numeric"] for r in records)
    verified_count = sum(r["verified_by_code"] for r in records)
    numeric_unverified = [r for r in records if r["is_numeric"] and not r["verified_by_code"]]
    derived_numeric_unverified = [r for r in numeric_unverified if r["status"] == "derived"]
    ambiguous = [r for r in records if r["status"] == "ambiguous"]
    empty_options = [r["id"] for r in records if not r["answer_options"]]

    lines = [
        f"records={len(records)}",
        f"overrides_applied={sum(r['id'] in overrides for r in records)}",
        "counts_by_week_status:",
    ]
    for week in range(1, 5):
        for status in ("derived", "ambiguous"):
            lines.append(f"  week={week} status={status} count={week_status[(week, status)]}")
    lines.append("counts_by_type_status:")
    for qtype in sorted({r.get("type", "unknown") for r in routes}):
        for status in ("derived", "ambiguous"):
            lines.append(f"  type={qtype} status={status} count={type_status[(qtype, status)]}")
    lines.extend([f"numeric_count={numeric_count}", f"verified_by_code_count={verified_count}",
                  f"numeric_derived_verified={sum(r['is_numeric'] and r['status']=='derived' and r['verified_by_code'] for r in records)}",
                  f"numeric_derived_unverified={len(derived_numeric_unverified)}"])
    lines.append("numeric_unverified_ids_and_reasons:")
    for r in numeric_unverified:
        lines.append(f"  {r['id']}: {r['verification_note']}")
    lines.append(f"ambiguous_count={len(ambiguous)}")
    lines.append("ambiguous_ids_and_notes:")
    for r in ambiguous:
        summary = (r["ambiguity_note"].replace("\n", " ").strip() or "No ambiguity note supplied.")
        lines.append(f"  {r['id']}: {summary}")
    lines.append(f"records_with_empty_answer_options={len(empty_options)}")
    for qid in empty_options:
        lines.append(f"  {qid}")
    lines.append(f"gate7_derived_numeric={len(derived_numeric_unverified)} unverified; each listed above with reason")
    lines.append(f"checker_match_count={sum(value == 'MATCH' for value in checked.values())}")
    lines.append(f"checker_mismatch_count={sum(value == 'MISMATCH' for value in checked.values())}")
    lines.append(override_verification_output.rstrip())
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
