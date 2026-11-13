"""Verify generated LLM exam banks, coverage, anchors, and linked images."""
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import unquote
import io
import json
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "sources/exam-papers"
INV = json.loads((BASE / "question-inventory.json").read_text())
ROUTING = json.loads((BASE / "routing.json").read_text())["records"]
SOLUTIONS = json.loads((BASE / "question-solutions.json").read_text())["records"]
TOPICS = json.loads((BASE.parent / "topics.json").read_text())["weeks"]
assert len(INV["questions"]) == len(ROUTING) == 488
assert len(SOLUTIONS) == 241

def run_checks():
    routed = [r for r in ROUTING if isinstance(r["week"], int) and 1 <= r["week"] <= 4]
    archived = [r for r in ROUTING if r["week"] == "archive"]
    assert len(routed) == 241, len(routed)
    route_ids = [r["id"] for r in ROUTING]
    assert len(route_ids) == len(set(route_ids)) == 488
    print(f"Inputs: {len(INV['questions'])} inventory records; {len(routed)} routed; {len(archived)} archived.")

    index_path = ROOT / "exam-question-index.md"
    index = index_path.read_text()
    bank_texts = {}
    anchors = []
    image_total = 0
    no_direct = []
    for week in range(1, 5):
        bank_path = ROOT / f"week {week}/week-{week}-exam-questions.md"
        text = bank_path.read_text()
        bank_texts[week] = text
        expected = {r["id"] for r in routed if r["week"] == week}
        actual_list = re.findall(r'<a id="([^"]+)"></a>', text)
        actual = set(actual_list)
        assert len(actual_list) == len(actual), f"Duplicate anchors in W{week}"
        actual_questions = {qid for qid in actual if qid.startswith("da5004_")}
        assert expected == actual_questions, (week, len(expected), len(actual_questions), expected - actual_questions, actual_questions - expected)
        assert text.count("#### Worked answer") == len(expected)
        assert all((f"<a id=\"{qid}\"></a>" in text and f"verified_by_code:**" in text[text.index(f'<a id="{qid}"></a>'):]) for qid in expected)
        print(f"Week {week}: {len(expected)} routed sections; IDs unique; each has worked answer and verification marker.")
        # Check every cell in the topic checklist table and ensure all syllabus topics occur as rows.
        marker = text.index("| Topic | Linked questions | Evidence / revision |")
        table = text[marker:].split("\n\n", 1)[0].splitlines()[2:]
        row_names = []
        for line in table:
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            assert len(cells) == 3 and all(cells), (week, line)
            row_names.append(re.sub(r"^<a id=\"[^\"]+\"></a>", "", cells[0]))
        names = {t["name"] for t in TOPICS[str(week)]["topics"]}
        assert names == set(row_names), (week, names - set(row_names), set(row_names) - names)
        route_topic_ids = {topic_id for r in routed if r["week"] == week for topic_id in r["topic_ids"]}
        for topic in TOPICS[str(week)]["topics"]:
            if topic["id"] not in route_topic_ids:
                no_direct.append((week, topic["id"], topic["name"]))
                assert f"No direct question found; revise the linked study notes." in text
        print(f"Week {week}: {len(names)} topic checklist rows; no blank cells; all topics.json topics present.")
        images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", text)
        for alt, url in images:
            assert alt.strip(), (bank_path, url, "image has no contextual alt text")
            decoded = unquote(url)
            target = (bank_path.parent / decoded.split("#", 1)[0]).resolve()
            assert target.is_file(), (bank_path, url, target)
        image_total += len(images)

    all_bank_ids = [qid for text in bank_texts.values() for qid in re.findall(r'<a id="(da5004_[^"]+)"></a>', text)]
    assert len(all_bank_ids) == len(set(all_bank_ids)) == 241
    assert set(all_bank_ids) == {r["id"] for r in routed}
    for record in archived:
        assert record["id"] in index, f"Archived ID absent from index: {record['id']}"
        assert all(record["id"] not in text for text in bank_texts.values()), f"Archived ID appeared in bank: {record['id']}"
    print(f"Archive: {len(archived)} IDs appear in the index and none appear in the four banks.")
    print(f"Images: {image_total} contextual image links resolve from their bank files.")
    counts = Counter(r["week"] for r in routed)
    assert sum(counts[w] for w in range(1, 5)) + len(archived) == 488
    assert f"routed W1 {counts[1]} + W2 {counts[2]} + W3 {counts[3]} + W4 {counts[4]} + archived {len(archived)}" in index
    print(f"Reconciliation: W1 {counts[1]} + W2 {counts[2]} + W3 {counts[3]} + W4 {counts[4]} + archived {len(archived)} = 488.")
    print("Topics without direct question: " + ("; ".join(f"W{w} {name} ({topic_id})" for w, topic_id, name in no_direct) if no_direct else "none"))
    print("PASS: exam bank structure, coverage, topic checklists, archive routing, and image links verified.")

capture = io.StringIO()
with redirect_stdout(capture):
    run_checks()
output = capture.getvalue()
print(output, end="")
(BASE / "bank-verify.txt").write_text(output)
