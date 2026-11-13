#!/usr/bin/env python3
"""Run the final Week 1–4 LLM vault gates and save an auditable transcript."""

from __future__ import annotations

import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "sources/gates-output.txt"
REPORT = ROOT / "sources/final-report-data.md"
EXAM = ROOT / "sources/exam-papers"
FENCE = re.compile(r"^\s{0,3}(```+|~~~+)(.*)$")


def run_cmd(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def normalize_stem(value: str) -> str:
    value = re.sub(r"!\[[^]]*\]\([^)]*\)", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", value.casefold())).strip()


def capture_gate(name: str, fn, out: list[str]) -> bool:
    buf = io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(out)
        if result is False:
            ok = False
    except Exception as exc:  # gates must not mask one another
        ok = False
        print(f"ERROR: {type(exc).__name__}: {exc}", file=buf)
    body = buf.getvalue().rstrip()
    out.append(f"\n{'=' * 78}\n{name}: {'PASS' if ok else 'FAIL'}\n{'=' * 78}\n{body or '(no output)'}")
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return ok


def g1(out):
    routing = load_json(EXAM / "routing.json")["records"]
    inventory = load_json(EXAM / "question-inventory.json")["questions"]
    index = (ROOT / "exam-question-index.md").read_text(encoding="utf-8")
    banks = {w: (ROOT / f"week {w}/week-{w}-exam-questions.md").read_text(encoding="utf-8") for w in range(1, 5)}
    counts = collections.Counter(r["week"] for r in routing)
    routed = sum(counts[w] for w in range(1, 5))
    archive = counts["archive"]
    all_ids = [r["id"] for r in routing]
    duplicated = sorted(q for q, n in collections.Counter(all_ids).items() if n != 1)
    bank_ids = {w: re.findall(r'<a id="(da5004_[^"]+)"></a>', text) for w, text in banks.items()}
    misplaced = []
    for w, ids in bank_ids.items():
        wanted = {r["id"] for r in routing if r["week"] == w}
        for q in sorted(set(ids) ^ wanted):
            misplaced.append(f"W{w}:{q}")
    missing_archive = [r["id"] for r in routing if r["week"] == "archive" and r["id"] not in index]
    inventory_by = {q["id"]: q for q in inventory}
    topic_by = {r["id"]: r for r in routing}
    print(f"routed={routed}; archived={archive}; total={routed + archive}; IDs={len(all_ids)}")
    print("| Week | Count | MCQ | MSQ | SA | Distinct stems |")
    print("|---:|---:|---:|---:|---:|---:|")
    for w in range(1, 5):
        rows = [r for r in routing if r["week"] == w]
        types = collections.Counter(r.get("type", "?") for r in rows)
        stems = {normalize_stem(inventory_by[r["id"]].get("stem_text", "")) for r in rows}
        print(f"| {w} | {len(rows)} | {types['MCQ']} | {types['MSQ']} | {types['SA']} | {len(stems)} |")
    print(f"Duplicate routing IDs: {len(duplicated)}" + (" — " + ", ".join(duplicated) if duplicated else ""))
    print(f"Bank/index placement discrepancies: {len(misplaced) + len(missing_archive)}")
    for item in misplaced:
        print("  " + item)
    for q in missing_archive:
        print(f"  archived ID absent from index: {q}")
    return routed + archive == 488 and len(all_ids) == 488 and not duplicated and not misplaced and not missing_archive


def g2(out):
    spec = importlib.util.spec_from_file_location("llm_check_links", ROOT / "scripts/check_links.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    files = module.markdown_files()
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        status = module.main()
    finally:
        sys.stdout = old_stdout
    print(buffer.getvalue().rstrip())
    text = buffer.getvalue()
    broken = re.search(r"Broken links \((\d+)\):", text)
    anchor_checks = 0
    for path in files:
        for _, line in module.active_lines(path.read_text(encoding="utf-8")):
            anchor_checks += sum("#" in m.group(1) for m in module.WIKI_RE.finditer(line))
            anchor_checks += sum("#" in m.group(2) for m in module.MD_RE.finditer(line))
    errors = text.split("Broken links (", 1)[1].split("\n", 1)[1] if "Broken links (" in text else ""
    counts = collections.Counter()
    for line in errors.splitlines():
        if " — " not in line:
            continue
        kind = line.split(" — ", 1)[1].split(" — ", 1)[0]
        if kind == "wikilink":
            counts["wikilinks"] += 1
        elif kind == "markdown-link":
            counts["markdown-links"] += 1
        elif kind == "image":
            counts["image-paths"] += 1
        if "anchor #" in line:
            counts["#anchors"] += 1
    print(f"Anchor references checked: {anchor_checks}")
    print("Broken by type: " + ", ".join(f"{k}={counts[k]}" for k in ("wikilinks", "markdown-links", "image-paths", "#anchors")))
    return status == 0 and broken is not None and broken.group(1) == "0" and len(files) > 0


def g3(out):
    p = run_cmd(["uv", "run", "python", "scripts/verify_exam_banks.py"])
    print(p.stdout.rstrip())
    good = p.returncode == 0
    topics = load_json(ROOT / "sources/topics.json")["weeks"]
    violations = []
    for week in range(1, 5):
        path = ROOT / f"week {week}/week-{week}-exam-questions.md"
        text = path.read_text(encoding="utf-8")
        marker = "| Topic | Linked questions | Evidence / revision |"
        if marker not in text:
            violations.append(f"week {week}: missing topic checklist table")
            continue
        table = text.split(marker, 1)[1].split("\n\n", 1)[0].splitlines()[1:]
        rows = []
        for line_no, line in enumerate(table, 1):
            if not line.startswith("|") or re.match(r"\|\s*:?-+", line):
                continue
            cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
            if len(cells) < 3 or any(not c.strip() for c in cells[:3]):
                violations.append(f"week {week} checklist row {line_no}: empty topic/table cell")
            elif not cells[0].strip().startswith("<a id="):
                rows.append(cells[0].strip())
        wanted = {topic["name"] for topic in topics[str(week)]["topics"]}
        absent = sorted(wanted - set(rows))
        if absent:
            violations.append(f"week {week}: missing topics {absent}")
        print(f"Independent W{week}: topic rows={len(rows)}, expected topics={len(wanted)}, missing={len(absent)}")
    print("Independent empty/missing-cell/topic findings:")
    print("\n".join(violations) if violations else "none")
    return good and not violations


def sha_map(week: int) -> dict[str, str]:
    base = ROOT / f"week {week}/assets"
    return {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*.png")) if "exam-papers" not in p.parts and p.is_file()}


def g4(out):
    output = []
    all_ok = True
    for week in range(1, 5):
        before_status = run_cmd(["git", "status", "--porcelain", "--", f"LLM/week {week}/assets"], cwd=REPO).stdout
        before = sha_map(week)
        p1 = run_cmd(["uv", "run", "python", f"generate_week{week}_plots.py"])
        after1 = sha_map(week)
        p2 = run_cmd(["uv", "run", "python", f"generate_week{week}_plots.py"])
        after2 = sha_map(week)
        status = run_cmd(["git", "status", "--porcelain", "--", f"LLM/week {week}/assets"], cwd=REPO).stdout
        same = after1 == after2
        unchanged = not before_status.strip() and not status.strip()
        print(f"W{week}: run1 exit={p1.returncode}; run2 exit={p2.returncode}; PNGs={len(after2)}; IDENTICAL={same}; committed-status-unchanged={unchanged}")
        if p1.stdout.strip():
            print(f"W{week} run 1 output:\n{p1.stdout.rstrip()}")
        if p2.stdout.strip():
            print(f"W{week} run 2 output:\n{p2.stdout.rstrip()}")
        output.append(f"Week {week}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(after2)} PNGs)\n"
                      "Run 1:\n" + "\n".join(f"{name}  {digest}" for name, digest in sorted(after1.items())) +
                      "\nRun 2:\n" + "\n".join(f"{name}  {digest}" for name, digest in sorted(after2.items())))
        if not same:
            changed = sorted(set(after1) | set(after2))
            for name in changed:
                if after1.get(name) != after2.get(name):
                    print(f"  hash differs: {name}: {after1.get(name)} -> {after2.get(name)}")
        if status.strip():
            print(f"  git status after generation:\n{status.rstrip()}")
        all_ok &= p1.returncode == 0 and p2.returncode == 0 and same and unchanged
    (ROOT / "plot_sha256.txt").write_text("\n\n".join(output) + "\n", encoding="utf-8")
    return all_ok


def fences(text: str):
    blocks, opened, body = [], None, []
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if not match:
            if opened:
                body.append(line)
            continue
        marker = match.group(1)
        lang = match.group(2).strip().split(maxsplit=1)[0].lower() if match.group(2).strip() else ""
        if opened is None:
            opened = (marker[0], len(marker), lang, number)
            body = []
        elif marker[0] == opened[0] and len(marker) >= opened[1]:
            blocks.append((opened[2], "\n".join(body), opened[3], number))
            opened, body = None, []
    return blocks


def g5(out):
    problems, count, placeholders = [], 0, []
    for week in range(1, 5):
        note = ROOT / f"week {week}/week-{week}-notes.md"
        cb = note.parent / "codeblocks"
        blocks = fences(note.read_text(encoding="utf-8"))
        python_blocks = [(i, b) for i, b in enumerate(blocks) if b[0] == "python"]
        files = sorted(cb.glob("[0-9][0-9]_*.py"))
        print(f"W{week}: note Python blocks={len(python_blocks)}; saved scripts={len(files)}")
        for _, (_, body, line, _) in python_blocks:
            matches = [p for p in files if p.read_text(encoding="utf-8").strip() == body.strip()]
            if len(matches) != 1:
                problems.append(f"{rel(note)}:{line}: Python block matches {len(matches)} saved .py files")
                continue
            script = matches[0]
            txt = script.with_suffix(".txt")
            next_block = next((b for b in blocks if b[2] > line), None)
            if not txt.is_file():
                problems.append(f"{rel(script)}: missing matching .txt")
            elif next_block is None or next_block[0] != "text" or next_block[1].strip() != txt.read_text(encoding="utf-8").strip():
                problems.append(f"{rel(note)}:{line}: following output block differs from {rel(txt)}")
            p = run_cmd(["uv", "run", "python", str(script.relative_to(ROOT))])
            count += 1
            if p.returncode or p.stdout.strip() != (txt.read_text(encoding="utf-8").strip() if txt.exists() else ""):
                problems.append(f"{rel(script)}: fresh stdout (exit {p.returncode}) differs from saved .txt")
            for offset, code_line in enumerate(body.splitlines(), 1):
                if re.search(r"\bTODO\b|pass\s+#|NotImplemented", code_line) or ("..." in code_line and not re.fullmatch(r"\s*(?:\.\.\.|Ellipsis)(?:\s*#.*)?", code_line)):
                    placeholders.append(f"{rel(note)}:{line + offset}: {code_line.strip()}")
        for script in files:
            for offset, code_line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"\bTODO\b|pass\s+#|NotImplemented", code_line) or ("..." in code_line and not re.fullmatch(r"\s*(?:\.\.\.|Ellipsis)(?:\s*#.*)?", code_line)):
                    placeholders.append(f"{rel(script)}:{offset}: {code_line.strip()}")
        for script in files:
            if not script.with_suffix(".txt").is_file():
                problems.append(f"{rel(script)}: missing output file")
    print(f"Codeblocks rerun: {count}")
    print(f"Placeholder findings: {len(placeholders)}")
    for item in placeholders:
        print("  " + item)
    print(f"Code/output findings: {len(problems)}")
    for item in problems:
        print("  " + item)
    return not problems and not placeholders


def g6(out):
    try:
        import yaml
        parse_frontmatter = lambda source: (yaml.safe_load(source), None)
        parser_name = "PyYAML"
    except ImportError:
        spec = importlib.util.spec_from_file_location("llm_notes_quality", ROOT / "scripts/check_notes_quality.py")
        checker = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(checker)
        parse_frontmatter = lambda source: checker.frontmatter("---\n" + source + "\n---")
        parser_name = "check_notes_quality.py YAML-subset parser"
    failures = []
    files = sorted(p for p in ROOT.rglob("*.md") if ".venv" not in p.parts and not ("sources" in p.parts and "ebook" in p.parts and "pages" in p.parts))
    fm_count = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines and lines[0].strip() == "---":
            end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
            if end is None:
                failures.append(f"{rel(path)}:1 unclosed frontmatter")
            else:
                fm_count += 1
                try:
                    _, parse_error = parse_frontmatter("\n".join(lines[1:end]))
                    if parse_error:
                        failures.append(f"{rel(path)}: frontmatter YAML parse error: {parse_error}")
                except Exception as exc:
                    failures.append(f"{rel(path)}: frontmatter YAML parse error: {exc}")
                for lineno, line in enumerate(lines[1:end], 2):
                    if "[[" in line and not re.search(r'"[^"\n]*\[\[[^]]+\]\][^"\n]*"', line):
                        failures.append(f"{rel(path)}:{lineno}: frontmatter wikilink not double-quoted")
        for lineno, line in enumerate(lines, 1):
            if line.lstrip().startswith("|") and "[[" in line:
                for match in re.finditer(r"\[\[([^]]+)\]\]", line):
                    if re.search(r"(?<!\\)\|", match.group(1)):
                        failures.append(f"{rel(path)}:{lineno}: unescaped wikilink pipe")
    print(f"Markdown files={len(files)}; frontmatter blocks parsed={fm_count} with {parser_name}; violations={len(failures)}")
    for item in failures:
        print("  " + item)
    return not failures


def g7(out):
    records = load_json(EXAM / "question-solutions.json")["records"]
    derived_numeric = [r for r in records if r.get("status") == "derived" and r.get("is_numeric") is True]
    violations = [r for r in derived_numeric if not r.get("verified_by_code") and not str(r.get("verification_note") or "").strip()]
    print(f"Derived numeric answers={len(derived_numeric)}; verified_by_code={sum(bool(r.get('verified_by_code')) for r in derived_numeric)}; reason-stated={len(derived_numeric)-sum(bool(r.get('verified_by_code')) for r in derived_numeric)}; violations={len(violations)}")
    for r in violations:
        print(f"  {r['id']}: verified_by_code={r.get('verified_by_code')}; verification_note={r.get('verification_note')!r}")
    return not violations


def g8(out):
    routing = load_json(EXAM / "routing.json")["records"]
    sol = load_json(EXAM / "question-solutions.json")["records"]
    topics = load_json(ROOT / "sources/topics.json")
    duplicate = lambda rows: sorted(q for q, n in collections.Counter(r["id"] for r in rows).items() if n > 1)
    dup_route, dup_sol = duplicate(routing), duplicate(sol)
    print(f"routing.json records={len(routing)}; question-solutions.json records={len(sol)}; topics.json parses=True")
    print(f"Duplicate routing IDs={len(dup_route)}; duplicate solution IDs={len(dup_sol)}; week topic groups={len(topics.get('weeks', {}))}")
    if dup_route or dup_sol:
        print("Duplicate IDs: " + ", ".join(dup_route + dup_sol))
    return len(routing) == 488 and len(sol) == 241 and len(topics.get("weeks", {})) == 4 and not dup_route and not dup_sol


def g9(out):
    status = run_cmd(["git", "status", "--porcelain", "LLM/"], cwd=REPO).stdout
    entries = [line for line in status.splitlines() if line.strip()]
    allowed_untracked = re.compile(r"^(?: M LLM/(scripts/run_gates\.py|sources/gates-output\.txt)|\?\? LLM/(scripts/run_gates\.py|plot_sha256\.txt|sources/(gates-output\.txt|final-report-data\.md|exam-papers/solutions-terra-w[12]\.json|ebook/(pages/.*\.png|week-[1-4]-transcript\.md|transcript-check\.txt))))$")
    unexpected = [line for line in entries if not allowed_untracked.match(line)]
    untracked = [line[3:] for line in entries if line.startswith("?? ")]
    print("git status --porcelain LLM/:")
    print(status.rstrip() if status.strip() else "(clean)")
    print(f"Untracked files={len(untracked)}")
    for path in untracked:
        print("  " + path)
    large = []
    for path in ROOT.rglob("*"):
        if ".venv" in path.parts:
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > 5 * 1024 * 1024 and path.suffix.lower() not in {".pdf", ".png"}:
            large.append((path.stat().st_size, rel(path)))
    largest = sorted(((p.stat().st_size, rel(p)) for p in ROOT.rglob("*") if ".venv" not in p.parts and p.is_file()), reverse=True)[:5]
    print(f"Files >5 MB outside PDF/PNG caches: {len(large)}")
    for size, path in large:
        print(f"  {size} bytes {path}")
    print("Five largest LLM files:")
    for size, path in largest:
        print(f"  {size} bytes {path}")
    return not unexpected and not large


def review_summary():
    result = collections.Counter()
    files = sorted((ROOT / "sources").glob("review-*.md"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        reviewer = "terra" if "terra" in path.name else "luna" if "luna" in path.name else "unknown"
        counts = re.search(r"Counts:\s*\*\*(\d+) errors,\s*(\d+) warnings,\s*(\d+) nits\*\*", text, re.I)
        if counts:
            for severity, value in zip(("error", "warning", "nit"), counts.groups()):
                result[(reviewer, severity)] += int(value)
        summary = re.search(r"Errors:\s*(\d+)\s*\nWarnings:\s*(\d+)\s*\nNits:\s*(\d+)\s*\nTotal findings:\s*(\d+)", text, re.I)
        if summary:
            for severity, value in zip(("error", "warning", "nit"), summary.groups()[:3]):
                result[(reviewer, severity)] += int(value)
        if "review-luna-routing.md" == path.name:
            result[(reviewer, "high-confidence-findings")] += len(re.findall(r"\*\*High\.\*\*", text))
            result[(reviewer, "medium-confidence-findings")] += len(re.findall(r"\*\*Medium\b", text))
        if path.name == "review-terra-ambiguous.md":
            total = re.search(r"\|\s*\*\*Total\*\*\s*\|\s*\*\*\d+\*\*\s*\|\s*\*\*\d+\*\*\s*\|\s*\*\*\d+\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|", text)
            if total:
                result[(reviewer, "unrated-ambiguity-cases-reviewed")] += int(total.group(1))
        if re.search(r"Total findings:\s*0", text, re.I) or re.search(r"found no substantive disagreement", text, re.I):
            result[(reviewer, "explicit-zero-finding-reviews")] += 1
    return files, result


def build_report():
    routing = load_json(EXAM / "routing.json")["records"]
    solutions = load_json(EXAM / "question-solutions.json")["records"]
    topics = load_json(ROOT / "sources/topics.json")["weeks"]
    inventory = {q["id"]: q for q in load_json(EXAM / "question-inventory.json")["questions"]}
    route_by_id = {r["id"]: r for r in routing}
    sol_by_id = {r["id"]: r for r in solutions}
    lines = ["# Final gate report data", "", "Generated from current LLM JSON, Markdown, scripts, and Git records.", "", "## (a) Topics by week", ""]
    for week in range(1, 5):
        lines += [f"### Week {week}", "", "| Topic ID | Topic | Ebook PDF pages | Routed occurrences |", "|---|---|---:|---:|"]
        for topic in topics[str(week)]["topics"]:
            occ = sum(1 for r in routing if r.get("week") == week and topic["id"] in r.get("topic_ids", []))
            pages = ", ".join(map(str, topic.get("pdf_pages", [])))
            lines.append(f"| `{topic['id']}` | {topic['name']} | {pages} | {occ} |")
    lines += ["", "## (b) Routing and answer status by week", "", "| Week | Routed | Distinct stems | MCQ | MSQ | SA | MCQ derived/ambiguous | MSQ derived/ambiguous | SA derived/ambiguous | Total derived/ambiguous |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for week in range(1, 5):
        rows = [r for r in routing if r.get("week") == week]
        types = collections.Counter(r.get("type", "SA") for r in rows)
        stems = {normalize_stem(inventory[r["id"]].get("stem_text", "")) for r in rows}
        cells = []
        total_derived = total_ambiguous = 0
        for typ in ("MCQ", "MSQ", "SA"):
            ids = [r["id"] for r in rows if r.get("type") == typ]
            d = sum(sol_by_id.get(q, {}).get("status") == "derived" for q in ids)
            a = sum(sol_by_id.get(q, {}).get("status") == "ambiguous" for q in ids)
            cells.append(f"{d}/{a}")
            total_derived += d
            total_ambiguous += a
        lines.append(f"| {week} | {len(rows)} | {len(stems)} | {types['MCQ']} | {types['MSQ']} | {types['SA']} | " + " | ".join(cells) + f" | {total_derived}/{total_ambiguous} |")
    lines += ["", "SA combines numerical and short-answer records. Answer-status counts are routed questions with a matching record in question-solutions.json; archived questions are excluded.", "", "## (c) Archive count by likely_later_week", "", "| Likely later week | Count |", "|---|---:|"]
    archive_counts = collections.Counter(str(r.get("likely_later_week") or "(unset)") for r in routing if r.get("week") == "archive")
    for later, count in sorted(archive_counts.items(), key=lambda kv: (kv[0] == "(unset)", kv[0])):
        lines.append(f"| {later} | {count} |")
    lines += ["", "## (d) Ambiguous answer IDs and notes", ""]
    for week in range(1, 5):
        rows = [s for s in solutions if s.get("status") == "ambiguous" and s.get("week") == week]
        lines += [f"### Week {week} ({len(rows)})", ""]
        for s in sorted(rows, key=lambda r: r["id"]):
            note = re.sub(r"\s+", " ", str(s.get("ambiguity_note") or "No ambiguity note recorded.")).strip()
            lines.append(f"- `{s['id']}` — {note}")
        lines.append("")
    discrepancies = (ROOT / "sources/ebook-discrepancies.md").read_text(encoding="utf-8")
    discrepancy_rows = len(re.findall(r"^\|\s*\d+\s*\|", discrepancies, re.M))
    review_files, review_counts = review_summary()
    lines += ["## (e) Evidence and overrides", "", f"- Ebook discrepancies logged: {discrepancy_rows} rows.", "- Review files: " + ", ".join(p.name for p in review_files) + ".", "- Review findings by reviewer/severity (counts use explicit review summaries; routing confidence findings are shown separately):"]
    for (reviewer, severity), count in sorted(review_counts.items()):
        if count:
            lines.append(f"  - {reviewer} / {severity}: {count}")
    for name, path in (("routing", EXAM / "routing-overrides.json"), ("solutions", EXAM / "solutions-overrides.json"), ("corrections", EXAM / "question-corrections.json")):
        data = load_json(path)
        entries = data.get("overrides", data.get("corrections", []))
        lines.append(f"- {name.title()} overrides/corrections applied: {len(entries)} records from `{path.name}`.")
    lines += ["", "## (f) Commits on branch since scaffold", "", "```text"]
    log = run_cmd(["git", "log", "--oneline", "1eea9bb..HEAD"], cwd=REPO)
    lines.extend(log.stdout.rstrip().splitlines())
    lines += ["```", "", "## (g) Exact rebuild commands", "", "```bash", "cd LLM", "uv run python scripts/build_inventory.py", "uv run python scripts/merge_routing.py", "uv run python scripts/merge_solutions.py", "uv run python scripts/build_exam_question_banks.py", "uv run python scripts/verify_exam_banks.py", "uv run python scripts/check_links.py", "uv run python scripts/check_notes_quality.py", "for n in 1 2 3 4; do uv run python generate_week${n}_plots.py; done", "uv run python scripts/run_gates.py", "```", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {rel(REPORT)}; ambiguous answer entries={sum(s.get('status') == 'ambiguous' for s in solutions)}; discrepancy rows={discrepancy_rows}; review files={len(review_files)}")


def main() -> int:
    if sys.argv[1:] not in ([], ["G9"]):
        print("Usage: uv run python scripts/run_gates.py [G9]")
        return 2
    only_g9 = sys.argv[1:] == ["G9"]
    output = ["LLM Weeks 1–4 final gates", f"Repository: {REPO}"]
    results = []
    gates = (("G1 routing/bank/index reconciliation", g1), ("G2 internal links", g2),
                     ("G3 topic checklists", g3), ("G4 deterministic plots", g4),
                     ("G5 saved executable codeblocks", g5), ("G6 frontmatter and table pipes", g6),
                     ("G7 numeric answer provenance", g7), ("G8 JSON integrity", g8),
                     ("G9 repository hygiene and file sizes", g9))
    if only_g9:
        gates = (gates[-1],)
    for name, fn in gates:
        results.append((name, capture_gate(name, fn, output)))
    if not only_g9:
        build_report()
    output.append("\nFINAL SUMMARY\n" + "\n".join(f"{'PASS' if ok else 'FAIL'} {name}" for name, ok in results))
    OUT.write_text("\n".join(output) + "\n", encoding="utf-8")
    print("\n".join(f"{'PASS' if ok else 'FAIL'} {name}" for name, ok in results))
    print(f"Saved detailed output to {rel(OUT)}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
