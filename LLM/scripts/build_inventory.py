#!/usr/bin/env python3
"""Build a searchable inventory from the curated DA5004 exam markdown."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "sources" / "exam-papers"
OUTPUT = PAPERS_DIR / "question-inventory.json"
REPORT = PAPERS_DIR / "inventory-report.txt"
PAPER_RE = re.compile(r"da5004_(\d{4})(T\d+)_(Q1|Q2|ET)_(FN|AN|NA)\.md$")
QUESTION_RE = re.compile(r"(?m)^### Question (\d+)\s*$")
SECTION_RE = re.compile(r"(?m)^### (?:Question \d+|Shared context .+)$")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
OPTION_RE = re.compile(r"(?m)^\*\*([A-Z])\.\*\*\s*$")


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.S)
    if not match:
        raise ValueError("missing YAML frontmatter")
    values = {}
    for line in match.group(1).splitlines():
        m = re.match(r"([A-Za-z0-9_-]+):\s*(.*?)\s*$", line)
        if m:
            values[m.group(1)] = m.group(2).strip('"\'')
    return values, text[match.end():]


def clean_markdown(text: str) -> str:
    """Remove image syntax/captions and structural metadata, retaining prose."""
    text = IMAGE_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        if re.match(r"^\*\*Figure\s+\d+\.\*\*", line):
            continue
        if re.match(r"^\*\*Shared context:\*\*", line):
            continue
        if re.match(r"^\*Source option ID:\*", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def image_refs(block: str) -> list[str]:
    refs = []
    for _, path in IMAGE_RE.findall(block):
        path = path.strip()
        # Markdown references are relative to the exam-papers directory.
        if path not in refs:
            refs.append(path)
    return refs


def parse_type_and_marks(block: str) -> tuple[str, int | float | None]:
    match = re.search(r"(?m)^\*\*Type:\*\*\s*(.*?)\s*$", block)
    raw = match.group(1) if match else "unknown"
    raw = re.sub(r"\s*·\s*\*\*Marks:\*\*.*$", "", raw).strip()
    raw = re.sub(r"\s*\*\*Marks:\*\*.*$", "", raw).strip()
    type_map = {
        "Multiple Choice Question": "MCQ",
        "Multiple Select Question": "MSQ",
        "Short Answer Question": "SA",
        "Numerical Answer Type": "NAT",
        "Numerical Answer": "NAT",
    }
    qtype = type_map.get(raw, raw)
    marks_match = re.search(r"\*\*Marks:\*\*\s*(\d+(?:\.\d+)?)", block)
    marks: int | float | None = None
    if marks_match:
        value = float(marks_match.group(1))
        marks = int(value) if value.is_integer() else value
    return qtype, marks


def parse_options(block: str) -> tuple[list[dict], str]:
    heading = re.search(r"(?m)^\*\*Options\*\*\s*$", block)
    if not heading:
        return [], block
    option_area = block[heading.end():]
    answer = re.search(r"(?m)^\*\*Correct answer:\*\*", option_area)
    if answer:
        option_area = option_area[:answer.start()]
    markers = list(OPTION_RE.finditer(option_area))
    options = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(option_area)
        part = option_area[marker.end():end]
        id_match = re.search(r"(?m)^\*Source option ID:\*\s*`([^`]+)`", part)
        option_images = image_refs(part)
        option_text = clean_markdown(part)
        option_text = re.sub(r"(?m)^\*\*Figure\s+\d+\.\*\*.*$", "", option_text).strip()
        options.append({
            "label": marker.group(1),
            "text": option_text,
            "option_id": id_match.group(1) if id_match else None,
            "images": option_images,
        })
    return options, block[:heading.start()]


def parse_question(block: str, number: int, line_number: int, paper: str,
                   year: int, term: str, exam: str, session: str) -> dict:
    id_match = re.search(r"(?m)^\*\*Question ID:\*\*\s*`([^`]+)`", block)
    qtype, marks = parse_type_and_marks(block)
    options, stem_source = parse_options(block)
    # Exclude heading, anchor, metadata, answer key, and PDF page annotation.
    stem_source = re.sub(r"\A.*?(?=\*\*Question ID:\*\*)", "", stem_source, flags=re.S)
    stem_source = re.sub(r"(?m)^\*\*Question ID:\*\*.*$", "", stem_source)
    stem_source = re.sub(r"(?m)^\*\*Portal container (?:number|ID):\*\*.*$", "", stem_source)
    stem_source = re.sub(r"(?m)^\*\*Type:\*\*.*$", "", stem_source)
    stem_source = re.sub(r"(?m)^\*\*PDF pages:\*\*.*$", "", stem_source)
    stem_source = re.sub(r"(?m)^\*\*Correct answer:\*\*.*$", "", stem_source)
    stem_source = re.sub(r"(?m)^\*\*Source evidence:\*\*.*$", "", stem_source)
    stem_source = re.sub(r"(?m)^<a id=.*?</a>\s*$", "", stem_source)
    stem_text = clean_markdown(stem_source)
    stem_images = image_refs(stem_source)
    # Shared-context link is metadata; any prose/math beneath it remains in stem.
    stem_text = re.sub(r"(?m)^\*\*Shared context:\*\*.*$", "", stem_text).strip()
    enough_text = len(re.sub(r"[^\w]", "", stem_text)) >= 35
    option_text_exists = any(re.sub(r"[^\w]", "", o["text"]) for o in options)
    image_only = (not enough_text and bool(stem_images)) or (not options) or (
        bool(options) and not option_text_exists and all(o["images"] for o in options)
    )
    return {
        "id": f"da5004_{paper}--q{number}",
        "paper": paper,
        "year": year,
        "term": term,
        "exam": exam,
        "session": session,
        "portal_question_id": id_match.group(1) if id_match else None,
        "number": number,
        "type": qtype,
        "marks": marks,
        "stem_text": stem_text,
        "stem_images": stem_images,
        "options": options,
        "image_only": image_only,
        "source_line": line_number,
    }


def main() -> int:
    paper_records = []
    questions = []
    discrepancies = []
    missing_images = []
    no_options = []
    unusual_types = []
    for path in sorted(PAPERS_DIR.glob("da5004_*.md")):
        match = PAPER_RE.fullmatch(path.name)
        if not match:
            continue
        year, term, exam, session = match.groups()
        paper = path.stem.removeprefix("da5004_")
        text = path.read_text(encoding="utf-8")
        meta, body = frontmatter(text)
        count = int(meta["academic_question_count"])
        scope_match = re.search(r"(?m)^\*\*Scope:\*\*\s*(.*)$", body)
        headings = list(QUESTION_RE.finditer(body))
        boundaries = list(SECTION_RE.finditer(body))
        parsed = []
        for i, heading in enumerate(headings):
            end = next((marker.start() for marker in boundaries
                        if marker.start() > heading.start()), len(body))
            block = body[heading.start():end]
            line = body.count("\n", 0, heading.start()) + 1 + text[:text.find(body)].count("\n")
            record = parse_question(block, int(heading.group(1)), line, paper,
                                    int(year), term, exam, session)
            parsed.append(record)
        paper_records.append({
            "paper": paper, "year": int(year), "term": term, "exam": exam,
            "session": session, "frontmatter_count": count,
            "parsed_count": len(parsed),
            "scope_note": scope_match.group(1).strip() if scope_match else "",
            "zero_questions": len(parsed) == 0,
        })
        if len(parsed) != count:
            discrepancies.append((paper, count, len(parsed)))
        for q in parsed:
            if not q["options"]:
                no_options.append(q["id"])
            if q["type"] not in {"MCQ", "MSQ", "SA", "NAT"}:
                unusual_types.append((q["id"], q["type"]))
            for image in q["stem_images"] + [im for opt in q["options"] for im in opt["images"]]:
                if not (PAPERS_DIR / image).is_file():
                    missing_images.append((q["id"], image))
            questions.append(q)

    ids = [q["id"] for q in questions]
    duplicate_ids = sorted(k for k, v in Counter(ids).items() if v > 1)
    payload = {"generated_by": "LLM/scripts/build_inventory.py", "papers": paper_records,
               "questions": questions}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["Exam paper question inventory verification", "", "paper | frontmatter | parsed",
             "--- | ---: | ---:"]
    lines.extend(f"{p['paper']} | {p['frontmatter_count']} | {p['parsed_count']}" for p in paper_records)
    lines += ["", f"Papers: {len(paper_records)}", f"Questions parsed: {len(questions)}",
              f"Frontmatter total: {sum(p['frontmatter_count'] for p in paper_records)}",
              f"Zero-question papers: {', '.join(p['paper'] for p in paper_records if p['zero_questions']) or 'none'}",
              f"Questions without options ({len(no_options)}): {', '.join(no_options) or 'none'}",
              f"Questions with unusual types ({len(unusual_types)}): " +
              (", ".join(f"{q}={t}" for q, t in unusual_types) or "none"),
              f"Duplicate IDs ({len(duplicate_ids)}): {', '.join(duplicate_ids) or 'none'}",
              f"Missing image references ({len(missing_images)}): " +
              (", ".join(f"{q}:{im}" for q, im in missing_images) or "none"),
              "", "Counts by type:"]
    lines.extend(f"{kind}: {n}" for kind, n in sorted(Counter(q["type"] for q in questions).items()))
    lines += [f"image_only records: {sum(q['image_only'] for q in questions)}", "",
              "Reconciliation differences:"]
    lines.extend(f"{p}: frontmatter declares {declared}; parsed {actual}" for p, declared, actual in discrepancies)
    if not discrepancies:
        lines.append("none")
    report_text = "\n".join(lines) + "\n"
    REPORT.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    errors = []
    if discrepancies:
        errors.append("per-paper count reconciliation failed")
    if len(questions) != 488:
        errors.append(f"expected 488 total questions, got {len(questions)}")
    if duplicate_ids:
        errors.append("duplicate question IDs")
    if missing_images:
        errors.append("missing referenced images")
    if errors:
        print("ASSERTION FAILURE: " + "; ".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
