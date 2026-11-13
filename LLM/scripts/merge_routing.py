#!/usr/bin/env python3
"""Merge reviewed routing parts and validate routing against the inventory."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "sources/exam-papers"
PARTS = [PAPERS / f"routing-part-{i}.json" for i in range(1, 5)]
INVENTORY = PAPERS / "question-inventory.json"
TOPICS = ROOT / "sources/topics.json"
OVERRIDES = PAPERS / "routing-overrides.json"
OUTPUT = PAPERS / "routing.json"
REPORT = PAPERS / "routing-report.txt"
META_FIELDS = ("paper", "year", "term", "exam", "session", "type", "marks", "number", "portal_question_id")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text: str) -> str:
    text = text.lower()
    # Ignore whitespace around TeX commands and common TeX spacing directives.
    text = re.sub(r"\\(?:[,;:!]|quad|qquad|enspace|hspace\{[^}]*\})", "", text)
    text = re.sub(r"\\(?:\s+)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def question_key(q: dict[str, Any]) -> tuple[str, ...]:
    opts = q.get("options") or []
    return (norm(q.get("stem_text") or ""), *(norm(o.get("text") or "") for o in opts))


def image_digest(q: dict[str, Any]) -> tuple[str, ...]:
    refs = list(q.get("stem_images") or [])
    for option in q.get("options") or []:
        refs.extend(option.get("images") or [])
    hashes = []
    for ref in refs:
        path = (PAPERS / ref).resolve()
        if path.is_file():
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            hashes.append(f"MISSING:{ref}")
    return tuple(sorted(hashes))


def main() -> int:
    inventory = load(INVENTORY)["questions"]
    by_id = {q["id"]: q for q in inventory}
    topics = load(TOPICS)["weeks"]
    topic_week = {topic["id"]: int(week) for week, data in topics.items() for topic in data["topics"]}
    topic_counts = Counter()
    part_records = []
    part_errors = []
    for path in PARTS:
        data = load(path)
        if not isinstance(data.get("records"), list):
            part_errors.append(f"{path.name}: records is not a list")
            continue
        part_records.extend(data["records"])
    part_ids = [r.get("id") for r in part_records]
    part_counts = Counter(part_ids)
    missing_parts = sorted(set(by_id) - set(part_ids))
    extra_parts = sorted(set(part_ids) - set(by_id))
    repeated_parts = sorted(k for k, v in part_counts.items() if v != 1)
    coverage_ok = len(inventory) == 488 and not part_errors and not missing_parts and not extra_parts and not repeated_parts and len(part_records) == 488

    records = []
    for raw in part_records:
        q = by_id.get(raw.get("id"))
        record = dict(raw)
        if q:
            for field in META_FIELDS:
                record[field] = q[field]
        records.append(record)

    override_data = load(OVERRIDES)
    overrides = override_data.get("overrides", [])
    override_errors = []
    by_record = {r.get("id"): r for r in records}
    for override in overrides:
        qid = override.get("id")
        if qid not in by_record:
            override_errors.append(f"unknown override ID: {qid}")
            continue
        record = by_record[qid]
        for field in ("week", "topic_ids", "archive_reason", "likely_later_week"):
            if field in override:
                record[field] = override[field]

    invalid_week = sorted(str(r.get("id")) for r in records if r.get("week") not in (1, 2, 3, 4, "archive"))
    topic_errors = []
    for r in records:
        week = r.get("week")
        ids = r.get("topic_ids") or []
        if week in (1, 2, 3, 4):
            if not ids:
                topic_errors.append(f"{r.get('id')}: no topic ids")
            for tid in ids:
                if tid not in topic_week:
                    topic_errors.append(f"{r.get('id')}: invalid topic id {tid}")
                elif topic_week[tid] != week:
                    topic_errors.append(f"{r.get('id')}: {tid} belongs to week {topic_week[tid]}, routed to {week}")
                else:
                    topic_counts[tid] += 1
    archive_errors = [str(r.get("id")) for r in records if r.get("week") == "archive" and (not str(r.get("archive_reason") or "").strip() or not r.get("likely_later_week"))]

    week_type = Counter((r.get("week"), r.get("type")) for r in records)
    week_exam = Counter((r.get("week"), r.get("exam")) for r in records)
    archive_later = Counter(str(r.get("likely_later_week")) for r in records if r.get("week") == "archive")
    low_conf = sum(str(r.get("confidence", "")).lower() == "low" for r in records)
    duplicates: dict[Any, dict[tuple[str, ...], list[str]]] = defaultdict(lambda: defaultdict(list))
    stem_seen: dict[Any, set[Any]] = defaultdict(set)
    for r in records:
        q = by_id.get(r.get("id"), {})
        key = question_key(q)
        if q.get("image_only"):
            digests = image_digest(q)
            # Without a resolvable image, there is no evidence that two
            # image-only records depict the same question.
            key = ("IMAGE_ONLY", *(digests or (str(r.get("id")),)))
        duplicates[r.get("week")][key].append(str(r.get("id")))
        stem_seen[r.get("week")].add(key)
    duplicate_groups = {w: sorted((ids for ids in groups.values() if len(ids) > 1), key=lambda ids: ids[0]) for w, groups in duplicates.items()}

    checks = [
        ("1 inventory coverage", coverage_ok, f"inventory={len(inventory)} routed={len(records)} missing={len(missing_parts)} extra={len(extra_parts)} repeated={len(repeated_parts)}"),
        ("2 allowed weeks", not invalid_week, f"invalid={len(invalid_week)}"),
        ("3 week-valid topics", not topic_errors, f"errors={len(topic_errors)}"),
        ("4 archive metadata", not archive_errors, f"errors={len(archive_errors)}"),
        ("5 routing overrides", not override_errors, f"applied={len(overrides)} errors={len(override_errors)}"),
    ]
    lines = ["Routing merge validation report", ""]
    for name, ok, detail in checks:
        lines.append(f"CHECK {name}: {'PASS' if ok else 'FAIL'} ({detail})")
    lines.append("Applied overrides:")
    for override in overrides:
        lines.append(f"  {override['id']}: week={override.get('week')} topics={','.join(override.get('topic_ids', []))}; {override.get('reason', '')}")
    lines.append("Rejected review suggestions:")
    for rejected in override_data.get("rejected", []):
        lines.append(f"  {rejected['id']}: {rejected.get('reason', '')}")
    if override_errors: lines.extend(["Override errors:", *["  " + e for e in override_errors]])
    if missing_parts: lines.append("Missing IDs: " + ", ".join(missing_parts))
    if extra_parts: lines.append("Extra IDs: " + ", ".join(extra_parts))
    if repeated_parts: lines.append("Repeated IDs: " + ", ".join(repeated_parts))
    if invalid_week: lines.append("Invalid-week IDs: " + ", ".join(invalid_week))
    if topic_errors: lines.extend(["Topic errors:", *["  " + e for e in topic_errors]])
    if archive_errors: lines.append("Archive metadata IDs: " + ", ".join(archive_errors))
    lines += ["", "Week × type:"]
    for week in (1, 2, 3, 4, "archive"):
        lines.append(f"  {week}: " + ", ".join(f"{typ}={week_type[(week, typ)]}" for typ in ("MCQ", "MSQ", "SA")))
    lines.append("Week × exam:")
    for week in (1, 2, 3, 4, "archive"):
        lines.append(f"  {week}: " + ", ".join(f"{exam}={week_exam[(week, exam)]}" for exam in ("Q1", "Q2", "ET")))
    lines.append("Topic occurrences:")
    for tid in topic_week:
        lines.append(f"  {tid}: {topic_counts[tid]}" + (" ZERO QUESTIONS" if topic_counts[tid] == 0 else ""))
    lines.append("Archive count by likely_later_week:")
    for later, count in sorted(archive_later.items()): lines.append(f"  {later}: {count}")
    lines.append(f"Low-confidence records: {low_conf}")
    lines.append("Distinct stems by week:")
    for week in (1, 2, 3, 4, "archive"):
        count = sum(r.get("week") == week for r in records)
        lines.append(f"  {week}: {len(stem_seen[week])} distinct / {count} records")
        for ids in duplicate_groups[week]: lines.append(f"    exact duplicate IDs: {', '.join(ids)}")

    # Preserve all parts; only write final artifacts after audit and only emit
    # the merged JSON when coverage is exact, so bad part data cannot be hidden.
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if coverage_ok:
        OUTPUT.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
