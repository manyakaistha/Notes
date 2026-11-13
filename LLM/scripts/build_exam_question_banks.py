"""Build the routed LLM exam banks and question index from reviewed JSON data.

Run from LLM/: ``uv run python scripts/build_exam_question_banks.py``.
"""
from collections import Counter
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "sources/exam-papers"
INV = json.loads((BASE / "question-inventory.json").read_text())
ROUTING = json.loads((BASE / "routing.json").read_text())["records"]
SOLUTIONS = json.loads((BASE / "question-solutions.json").read_text())["records"]
TOPICS = json.loads((BASE.parent / "topics.json").read_text())["weeks"]
Q_BY_ID = {q["id"]: q for q in INV["questions"]}
R_BY_ID = {q["id"]: q for q in ROUTING}
S_BY_ID = {q["id"]: q for q in SOLUTIONS}
PAPERS = {p["paper"]: p for p in INV["papers"]}
assert len(Q_BY_ID) == 488 and len(R_BY_ID) == 488 and len(S_BY_ID) == 241
assert set(Q_BY_ID) == set(R_BY_ID)
assert set(S_BY_ID) == {r["id"] for r in ROUTING if isinstance(r["week"], int)}

def wlink(target, label):
    return f"[[{target}\\|{label}]]"

def clean_text(value):
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()

def occurrence_label(q):
    exam = {"Q1": "Quiz 1", "Q2": "Quiz 2", "ET": "End Term"}[q["exam"]]
    return f"{q['year']} {q['term']} {exam} {q['session']}, Q{q['number']}"

def img_lines(paths, q, role):
    result = []
    for image in paths:
        rel = Path("../sources/exam-papers") / image
        target = (ROOT / f"week {R_BY_ID[q['id']]['week']}" / rel).resolve()
        assert target.is_file(), f"Missing source image: {target}"
        alt = f"{role} image for {occurrence_label(q)} in {q['paper']}"
        result.append(f"![{alt}]({rel.as_posix()})")
    return result

def topic_anchor_link(q):
    week = R_BY_ID[q["id"]]["week"]
    return wlink(f"week-{week}-exam-questions#{q['id']}", f"{q['year']}{q['term']} {q['exam']} {q['session']} Q{q['number']}")

def render_question(q, s):
    topics = R_BY_ID[q["id"]]["topic_ids"]
    stem_text = clean_text(q["stem_text"])
    out = [f'\n<a id="{q["id"]}"></a>\n', f"### {occurrence_label(q)}", "", f"**Paper:** `{q['paper']}` · **Type:** {q['type']} · **Marks:** {q['marks'] if q['marks'] is not None else 'not stated'} · **Topic(s):** {', '.join(topics) or 'No topic metadata'}", "", stem_text or "[No text extracted; inspect the source image.]", ""]
    out.extend(img_lines(q["stem_images"], q, "Question stem"))
    for option in q["options"]:
        out.extend(["", f"**{option['label']}.** {clean_text(option['text'])}"])
        out.extend(img_lines(option["images"], q, f"Option {option['label']}"))
    out.extend(["", "#### Worked answer", "", f"**Final answer:** {s['answer']}", "", f"**Steps:** {s['steps']}", "", f"**Status:** {s['status']}"])
    if s.get("ambiguity_note"):
        out.extend(["", f"**Ambiguity note:** {s['ambiguity_note']}"])
    out.extend(["", f"**verified_by_code:** `{str(s.get('verified_by_code') is True).lower()}` — {s.get('verification_note') or 'No verification note supplied.'}", "", "---"])
    return "\n".join(out)

included = {w: [Q_BY_ID[r["id"]] for r in ROUTING if r["week"] == w] for w in range(1, 5)}
for w, questions in included.items():
    folder = ROOT / f"week {w}"
    current_topics = TOPICS[str(w)]["topics"]
    topic_rows = []
    for topic in current_topics:
        matches = [q for q in questions if topic["id"] in R_BY_ID[q["id"]]["topic_ids"]]
        if matches:
            refs = " · ".join(topic_anchor_link(q) for q in matches)
        else:
            refs = f"No direct question found; revise the linked study notes. {wlink(f'week-{w}-notes', 'Week notes')}"
        topic_rows.append(f"| <a id=\"{topic['id']}\"></a>{topic['name']} | {len(matches)} | {refs} |")
    type_counts = Counter(q["type"] for q in questions)
    exam_counts = Counter(q["exam"] for q in questions)
    prev_bank = f'"[[week-{w-1}-exam-questions|Week {w-1} bank]]"' if w > 1 else '"[[Knowledge-Map|Knowledge Map]]"'
    next_bank = f'"[[week-{w+1}-exam-questions|Week {w+1} bank]]"' if w < 4 else '"[[Knowledge-Map|Knowledge Map]]"'
    prev_note = f'"[[week-{w-1}-notes|Week {w-1} notes]]"' if w > 1 else '"[[Knowledge-Map|Knowledge Map]]"'
    next_note = f'"[[week-{w+1}-notes|Week {w+1} notes]]"' if w < 4 else '"[[Knowledge-Map|Knowledge Map]]"'
    lines = [
        "---", f'title: "Week {w}: LLM Exam Questions and Worked Answers"', f"week: {w}", 'status: complete', 'date: 2026-11-13',
        f"prev: {prev_bank}", f"next: {next_bank}", f"notes: \"[[week-{w}-notes|Week {w} notes]]\"", 'map: "[[Knowledge-Map]]"',
        "tags:", f"  - llm/week-0{w}", "  - llm/exam-questions", "---", "", f"# Week {w}: Exam Questions and Worked Answers", "",
        f"**Navigation:** {wlink(f'week-{w-1}-exam-questions' if w > 1 else 'Knowledge-Map', f'Previous bank' if w > 1 else 'Knowledge Map')} · {wlink(f'week-{w}-notes', f'Week {w} notes')} · {wlink('Knowledge-Map', 'Knowledge Map')} · {wlink(f'week-{w+1}-exam-questions' if w < 4 else 'Knowledge-Map', f'Next bank' if w < 4 else 'Knowledge Map')} · {wlink(f'week-{w-1}-notes' if w > 1 else 'Knowledge-Map', 'Previous notes')} · {wlink(f'week-{w+1}-notes' if w < 4 else 'Knowledge-Map', 'Next notes')}", "",
        f"**Coverage:** {len(questions)} routed occurrences; {len({q['stem_text'].strip() for q in questions})} distinct stems; by type MCQ {type_counts['MCQ']}, MSQ {type_counts['MSQ']}, SA {type_counts['SA']}; by exam Q1 {exam_counts['Q1']}, Q2 {exam_counts['Q2']}, ET {exam_counts['ET']}.", "",
        "**Answer provenance:** Answers were derived by the study team (Terra/Luna agents); they are not official keys. `verified_by_code: true` means a numerical/code-based check was run for that answer; `false` means no such code verification is claimed. Status and any source ambiguity are stated with each solution.", "",
        "## Topic checklist", "", "The checklist includes every topic named for this week in `sources/topics.json`. A topic without a directly routed question still links to the study notes.", "",
        "| Topic | Linked questions | Evidence / revision |", "| :--- | ---: | :--- |", *topic_rows, "", "## Question navigator", "", "| Paper | Question link | Topic | Answer status |", "| :--- | :--- | :--- | :--- |"
    ]
    for q in questions:
        s = S_BY_ID[q["id"]]
        question_link = wlink(f"week-{w}-exam-questions#{q['id']}", occurrence_label(q))
        route_topics = R_BY_ID[q["id"]]["topic_ids"]
        lines.append(f"| {q['paper']} | {question_link} | {', '.join(route_topics) or 'No topic metadata'} | {s['status']} |")
    for exam, heading in (("Q1", "Quiz 1"), ("Q2", "Quiz 2"), ("ET", "End Term")):
        qs = [q for q in questions if q["exam"] == exam]
        if qs:
            lines.extend(["", f"## {heading} questions"])
            lines.extend(render_question(q, S_BY_ID[q["id"]]) for q in qs)
    lines.extend(["", "## Source and reproducibility", "", "Questions preserve each routed paper occurrence, including repeated stems from different papers or sessions. Images are linked from the archived extraction assets. Rebuild with `uv run python scripts/build_exam_question_banks.py`; run `uv run python scripts/build_exam_study_priorities.py` for the ranked study guide."])
    (folder / f"week-{w}-exam-questions.md").write_text("\n".join(lines) + "\n")

# Reconciled index: routed per-paper counts and archived records grouped by likely later week/reason.
index = ["---", 'title: "LLM Exam Question Index — Weeks 1–4"', 'status: complete', 'date: 2026-11-13', 'map: "[[Knowledge-Map]]"', 'tags: [llm/exam-questions, llm/moc]', "---", "", "# LLM Exam Question Index — Weeks 1–4", "", f"**Navigation:** {wlink('Knowledge-Map', 'Knowledge Map')} · {wlink('exam-study-guide', 'Exam study guide')}", "", "Each routed row represents one paper occurrence; repeated questions are intentionally retained. Archive entries remain indexed here with their routing reason.", "", "## Weekly banks", "", "| Week | Bank | Routed occurrences |", "| :--- | :--- | ---: |"]
for w in range(1, 5):
    index.append(f"| {w} | {wlink(f'week-{w}-exam-questions', f'Week {w} questions and worked answers')} | {len(included[w])} |")
index.extend(["", "## Paper counts", "", "| Paper | W1 | W2 | W3 | W4 | Archived | Total |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"])
for paper in sorted(PAPERS):
    rows = [r for r in ROUTING if r["paper"] == paper]
    byweek = [sum(r["week"] == w for r in rows) for w in range(1, 5)]
    archived = sum(r["week"] == "archive" for r in rows)
    index.append(f"| {paper} | {' | '.join(map(str, byweek))} | {archived} | {len(rows)} |")
index.extend(["", "## Archived questions grouped by likely later week and reason", "", "Archive records have no Week 1–4 solution bank entry. Their reasons and IDs are retained below."])
archived_records = [r for r in ROUTING if r["week"] == "archive"]
groups = {}
for r in archived_records:
    later = str(r.get("likely_later_week") or "Unspecified")
    reason = r.get("archive_reason") or "No archive reason supplied"
    groups.setdefault((later, reason), []).append(r)
for later, reason in sorted(groups, key=lambda x: (int(x[0]) if x[0].isdigit() else 999, x[1])):
    records = groups[(later, reason)]
    index.extend(["", f"### Likely later week {later} — {reason} ({len(records)})", "", ", ".join(f"`{r['id']}`" for r in records)])
routed_total = sum(map(len, included.values()))
index.extend(["", f"**Total reconciliation:** routed W1 {len(included[1])} + W2 {len(included[2])} + W3 {len(included[3])} + W4 {len(included[4])} + archived {len(archived_records)} = {routed_total} + {len(archived_records)} = {len(ROUTING)}.", ""])
(ROOT / "exam-question-index.md").write_text("\n".join(index))
print(f"Generated question banks: " + ", ".join(f"W{w}={len(included[w])}" for w in range(1, 5)) + f"; routed={routed_total}; archived={len(archived_records)}; total={len(ROUTING)}.")
