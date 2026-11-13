#!/usr/bin/env python3
"""Static quality checks for LLM weekly, concept, and question-bank notes."""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEEK_NOTE = re.compile(r"week (\d+)/week-\d+-notes\.md$")
EXAM_NOTE = re.compile(r"week (\d+)/week-\d+-exam-focused-notes\.md$")
FENCE = re.compile(r"^\s{0,3}(```+|~~~+)(.*)$")
IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(?:<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\s*\)")


def selected_files() -> list[Path]:
    found = []
    for p in ROOT.rglob("*.md"):
        if ".venv" in p.parts or ("sources" in p.parts and "ebook" in p.parts and "pages" in p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if WEEK_NOTE.search(rel) or EXAM_NOTE.search(rel) or rel.startswith("notes/concepts/"):
            found.append(p)
        elif "bank" in rel.casefold() or re.search(r"week-\d+-exam-questions\.md$", rel):
            found.append(p)
    return sorted(set(found))


def frontmatter(text: str) -> tuple[dict[str, object], str | None]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, "missing opening --- delimiter"
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, "missing closing --- delimiter"
    block = lines[1:end]
    if importlib.util.find_spec("yaml") is not None:
        import yaml  # type: ignore[import-not-found]
        try:
            data = yaml.safe_load("\n".join(block))
            if not isinstance(data, dict):
                return {}, "frontmatter must be a YAML mapping"
            return data, None
        except Exception as exc:  # retain parser diagnostics for the report
            return {}, f"YAML parse error: {exc}"

    # Validate the simple YAML mapping/list subset used by this vault.
    data: dict[str, object] = {}
    active_list: str | None = None
    for offset, raw in enumerate(block, 2):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("  - "):
            if active_list is None:
                return {}, f"line {offset}: list item without a mapping key"
            value = raw[4:].strip()
            if value.startswith(('"', "'")) and (len(value) < 2 or value[-1] != value[0]):
                return {}, f"line {offset}: unterminated quoted list value"
            if not isinstance(data[active_list], list):
                return {}, f"line {offset}: invalid list indentation"
            data[active_list].append(value.strip('"\''))
            continue
        match = re.match(r"^([A-Za-z_][\w-]*):(?:\s*(.*))?$", raw)
        if not match:
            return {}, f"line {offset}: unsupported or malformed YAML mapping entry"
        key, value = match.group(1), (match.group(2) or "").strip()
        if key in data:
            return {}, f"line {offset}: duplicate key {key}"
        active_list = None
        if value == "":
            data[key] = []
            active_list = key
        elif value.startswith("["):
            if not value.endswith("]"):
                return {}, f"line {offset}: unterminated inline list"
            body = value[1:-1].strip()
            # Split commas outside single/double quoted scalars.
            values, start, quote = [], 0, None
            for idx, char in enumerate(body):
                if quote:
                    if char == quote and (idx == 0 or body[idx - 1] != "\\"):
                        quote = None
                elif char in "\"'":
                    quote = char
                elif char == ",":
                    values.append(body[start:idx].strip())
                    start = idx + 1
            if quote:
                return {}, f"line {offset}: unterminated quoted inline-list value"
            if body:
                values.append(body[start:].strip())
            data[key] = [v.strip('"\'') for v in values]
        else:
            if value.startswith(('"', "'")) and (len(value) < 2 or value[-1] != value[0]):
                return {}, f"line {offset}: unterminated quoted scalar"
            data[key] = value.strip('"\'')
    return data, None


def fences(text: str) -> list[tuple[str, str, int, int]]:
    """Return (language, body, opening line, closing line) fenced blocks."""
    blocks = []
    opened = None
    body: list[str] = []
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
            opened = None
            body = []
    return blocks


def quality_paths(path: Path) -> tuple[str, ...]:
    failures = []
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT).as_posix()

    if text.startswith("---"):
        meta, error = frontmatter(text)
        if error:
            failures.append(f"frontmatter: {error}")
        else:
            required = ["title"]
            if WEEK_NOTE.search(rel) or EXAM_NOTE.search(rel) or "bank" in rel.casefold() or re.search(r"week-\d+-exam-questions\.md$", rel):
                required.extend(["week", "status"])
            missing = [key for key in required if key not in meta]
            if missing:
                failures.append("frontmatter: missing keys " + ", ".join(missing))
    else:
        failures.append("frontmatter: missing frontmatter")

    lines = text.splitlines()
    image_numbers = []
    for idx, line in enumerate(lines):
        if IMAGE.search(line):
            next_idx = idx + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            caption = re.match(r"^\*\*Figure\s+(\d+)\.\*\*", lines[next_idx].strip()) if next_idx < len(lines) else None
            if not caption:
                failures.append(f"figure: image at line {idx + 1} lacks immediate numbered caption")
            else:
                image_numbers.append((idx + 1, int(caption.group(1))))
    for expected, (line_no, number) in enumerate(image_numbers, 1):
        if number != expected:
            failures.append(f"figure: caption at line {line_no} is Figure {number}; expected Figure {expected}")

    is_week_note = bool(WEEK_NOTE.search(rel))
    if is_week_note:
        codeblocks = path.parent / "codeblocks"
        blocks = fences(text)
        python_blocks = [(i, block) for i, block in enumerate(blocks) if block[0] == "python"]
        py_files = {p.name: p for p in codeblocks.glob("[0-9][0-9]_*.py")}
        for block_index, (lang, body, line_no, _) in python_blocks:
            matches = [p for p in py_files.values() if p.read_text(encoding="utf-8").strip() == body.strip()]
            if len(matches) != 1:
                failures.append(f"code-match: Python block at line {line_no} matches {len(matches)} saved .py files")
                matched = None
            else:
                matched = matches[0]
            following_block = next((b for b in blocks if b[2] > line_no), None)
            if matched is None:
                continue
            expected_txt = matched.with_suffix(".txt")
            if not expected_txt.exists():
                failures.append(f"code-output: {matched.name} has no matching .txt")
            elif following_block is None or following_block[0] != "text":
                failures.append(f"code-output: block at line {line_no} is not followed by a ```text block")
            elif following_block[1].strip() != expected_txt.read_text(encoding="utf-8").strip():
                failures.append(f"code-output: text block after line {line_no} differs from {expected_txt.name}")
        for _, (_, body, opening, _) in python_blocks:
            for offset, code_line in enumerate(body.splitlines(), 1):
                if re.search(r"\bTODO\b|pass\s+#|NotImplemented", code_line) or ("..." in code_line and not re.fullmatch(r"\s*(?:\.\.\.|Ellipsis)(?:\s*#.*)?", code_line)):
                    failures.append(f"placeholder: Python block line {opening + offset}: {code_line.strip()}")

    if rel.startswith("notes/concepts/"):
        definition = re.search(r"\*\*Formal definition\.\*\*\s*(.*?)(?:\n\s*\n|\n\*\*)", text, re.S)
        count = len(re.findall(r"\b[\w’'-]+\b", definition.group(1))) if definition else 0
        if not definition:
            failures.append("concept: missing Formal definition paragraph")
        elif not 100 <= count <= 250:
            failures.append(f"concept: definition has {count} words (required 100–250)")
        if "**Appears in:**" not in text:
            failures.append("concept: missing **Appears in:**")
        if "## Related" not in text:
            failures.append("concept: missing ## Related")
        if "Why it matters here" not in text:
            failures.append("concept: missing Why it matters here")
        map_text = (ROOT / "Knowledge-Map.md").read_text(encoding="utf-8")
        if f"[[{path.stem}]]" not in map_text and f"[[{path.stem}|" not in map_text:
            failures.append("concept: not linked from Knowledge-Map.md")
    return tuple(failures)


def main() -> int:
    files = selected_files()
    rows = [(p, quality_paths(p)) for p in files]
    totals: Counter[str] = Counter()
    print("| File | Result | Findings |")
    print("|---|---:|---|")
    for path, failures in rows:
        rel = path.relative_to(ROOT).as_posix()
        print(f"| `{rel}` | {'FAIL' if failures else 'PASS'} | {'; '.join(failures) if failures else '—'} |")
        for finding in failures:
            category = finding.split(":", 1)[0]
            totals[category] += 1
    print(f"\nFiles checked: {len(files)}")
    print(f"Files with failures: {sum(bool(f) for _, f in rows)}")
    print("Failures by category: " + (", ".join(f"{k}={totals[k]}" for k in sorted(totals)) if totals else "none"))
    print(f"Total findings: {sum(totals.values())}")
    return 1 if totals else 0


if __name__ == "__main__":
    raise SystemExit(main())
