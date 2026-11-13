"""Build the occurrence-weighted topic study guide from routed records.

Run from LLM/: ``uv run python scripts/build_exam_study_priorities.py``.
"""
from collections import defaultdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "sources/exam-papers"
ROUTING = json.loads((BASE / "routing.json").read_text())["records"]
TOPICS = json.loads((BASE.parent / "topics.json").read_text())["weeks"]

def wlink(target, label):
    return f"[[{target}\\|{label}]]"

metrics = defaultdict(lambda: {"occurrences": 0, "marks": 0})
for record in ROUTING:
    if not isinstance(record["week"], int):
        continue
    for topic_id in record["topic_ids"]:
        item = metrics[topic_id]
        item["occurrences"] += 1
        item["marks"] += record["marks"] if isinstance(record.get("marks"), (int, float)) else 0

topic_records = []
for week_key, data in TOPICS.items():
    week = int(week_key)
    for topic in data["topics"]:
        m = metrics[topic["id"]]
        mean_marks = m["marks"] / m["occurrences"] if m["occurrences"] else 0
        score = m["occurrences"] * mean_marks
        topic_records.append({"week": week, "id": topic["id"], "name": topic["name"], "occurrences": m["occurrences"], "marks": m["marks"], "score": score})

out = [
    "---", 'title: "LLM Exam Study Guide — Weeks 1–4"', 'status: complete', 'date: 2026-11-13',
    'map: "[[Knowledge-Map]]"', 'tags: [llm/exam-study, llm/moc]', "---", "",
    "# LLM Exam Study Guide — Weeks 1–4", "",
    f"**Navigation:** {wlink('Knowledge-Map', 'Knowledge Map')} · {wlink('exam-question-index', 'Question index')}", "",
    "## Frequency and marks ranking", "",
    "For each topic, **score = occurrences × mean marks per occurrence = total routed marks** across its directly assigned questions. Occurrences are repeated paper records, so an item repeated across papers contributes repeatedly. This is a descriptive review-priority measure for the supplied corpus, not a prediction of future exam marks; questions may carry multiple topic IDs.", "",
    "| Rank | Week | Topic ID and name | Occurrences | Total marks | Score | Bank checklist | Study notes |", "| ---: | ---: | :--- | ---: | ---: | ---: | :--- | :--- |"
]
for rank, topic in enumerate(sorted(topic_records, key=lambda x: (-x["score"], -x["occurrences"], x["week"], x["name"])), 1):
    week = topic["week"]
    bank = wlink(f"week-{week}-exam-questions#{topic['id']}", "Checklist")
    notes = wlink(f"week-{week}-notes", f"Week {week} notes")
    out.append(f"| {rank} | {week} | `{topic['id']}` — {topic['name']} | {topic['occurrences']} | {topic['marks']} | {topic['score']:g} | {bank} | {notes} |")

out.extend(["", "## Reading order by week"])
for week_key, data in TOPICS.items():
    week = int(week_key)
    names = " → ".join(topic["name"] for topic in data["topics"])
    out.extend(["", f"**Week {week}.** Read the linked notes in their syllabus order: {names}. Then use the linked question bank checklist to identify direct evidence and work the routed occurrences in paper order. Topics without direct evidence remain part of the curriculum and should still be revised."])
out.append("")
(ROOT / "exam-study-guide.md").write_text("\n".join(out))
print(f"Generated study guide: {len(topic_records)} topics ranked; {sum(x['occurrences'] for x in topic_records)} topic-question assignments; total score {sum(x['score'] for x in topic_records):g}.")
