#!/usr/bin/env python3
"""Check local Markdown links in the LLM Obsidian vault."""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PENDING = {f"week-{n}-exam-questions" for n in range(1, 5)} | {
    "exam-question-index", "exam-study-guide", "week-n-exam-questions"
}
FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
WIKI_RE = re.compile(r"(?<!\\)\[\[([^\]]+?)\]\]")
MD_RE = re.compile(r"!?\[([^\]]*)\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\s*\)")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
ANCHOR_RE = re.compile(r"<a\s+[^>]*?id=[\"']([^\"']+)[\"'][^>]*>", re.I)


def markdown_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.md")
        if ".venv" not in p.parts and not ("sources" in p.parts and "ebook" in p.parts and "pages" in p.parts)
    )


def active_lines(text: str) -> list[tuple[int, str]]:
    """Return lines outside fenced code, preserving frontmatter for link checks."""
    result = []
    fence_char = None
    fence_len = 0
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE_RE.match(line)
        if match:
            mark = match.group(1)
            char, length = mark[0], len(mark)
            if fence_char is None:
                fence_char, fence_len = char, length
            elif char == fence_char and length >= fence_len:
                fence_char = None
            continue
        if fence_char is None:
            result.append((number, line))
    return result


def slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\-\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    found = set(ANCHOR_RE.findall(text))
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = slug(match.group(2))
        counts[base] += 1
        found.add(base if counts[base] == 1 else f"{base}-{counts[base]}")
        # Obsidian heading links use the visible heading text; Docsify uses its
        # slug. Accept both while keeping HTML IDs for stable question records.
        found.add(match.group(2).strip().rstrip("#").strip())
    return found


def outside_fence_text(text: str) -> list[tuple[int, str]]:
    return active_lines(text)


def resolve_wiki(target: str, source: Path, by_stem: dict[str, list[Path]]) -> tuple[bool, str]:
    target = target.strip()
    if not target:
        candidates = [source]
        fragment = ""
    else:
        raw_path, sep, fragment = target.partition("#")
        name = Path(raw_path).name
        if name.lower().endswith(".md"):
            name = name[:-3]
        candidates = by_stem.get(name.casefold(), [])
        if not candidates:
            return False, "target file does not exist"
    if sep and fragment:
        fragment = unquote(fragment)
        if not any(fragment in anchors(candidate) for candidate in candidates):
            return False, f"anchor #{fragment} does not exist"
    return True, ""


def resolve_md(target: str, source: Path) -> tuple[bool, str]:
    target = unquote(target.strip().strip("<>"))
    parts = urlsplit(target)
    if parts.scheme.lower() in {"http", "https", "mailto"}:
        return True, ""
    if parts.scheme:
        return True, ""
    path_part = parts.path
    destination = (source.parent / path_part).resolve() if path_part else source.resolve()
    if path_part and not destination.exists():
        return False, "target path does not exist"
    if parts.fragment:
        fragment = unquote(parts.fragment)
        if not destination.is_file() or destination.suffix.lower() != ".md":
            return False, f"fragment #{fragment} requires a Markdown file"
        if fragment not in anchors(destination):
            return False, f"anchor #{fragment} does not exist"
    return True, ""


def is_pending(target: str, by_stem: dict[str, list[Path]]) -> bool:
    raw = target.split("#", 1)[0].strip()
    name = Path(raw).name
    if name.lower().endswith(".md"):
        name = name[:-3]
    return name.casefold() in PENDING and not by_stem.get(name.casefold())


def main() -> int:
    files = markdown_files()
    by_stem: dict[str, list[Path]] = defaultdict(list)
    cache: dict[Path, str] = {}
    for path in files:
        by_stem[path.stem.casefold()].append(path)
        cache[path] = path.read_text(encoding="utf-8")
    counts: Counter[str] = Counter()
    errors: list[tuple[Path, int, str, str, str]] = []
    pending: list[tuple[Path, int, str, str]] = []

    for path in files:
        text = cache[path]
        lines = outside_fence_text(text)
        frontmatter_open = bool(text.startswith("---\n") or text.startswith("---\r\n"))
        frontmatter_end = next((n for n, line in enumerate(text.splitlines(), 1) if n > 1 and line.strip() == "---"), None) if frontmatter_open else None
        table_lines = {n for n, line in lines if "|" in line and line.strip().startswith("|")}
        for line_no, line in lines:
            for match in WIKI_RE.finditer(line):
                spec = match.group(1).strip()
                separator = re.search(r"(?<!\\)\|", spec)
                alias_split = re.search(r"\\?\|", spec)
                target = (spec[:alias_split.start()] if alias_split else spec).strip().rstrip("\\")
                counts["wikilinks"] += 1
                ok, reason = resolve_wiki(target, path, by_stem)
                if not ok:
                    if is_pending(target, by_stem):
                        pending.append((path, line_no, "wikilink", target))
                    else:
                        errors.append((path, line_no, "wikilink", target, reason))
                if line_no in table_lines and separator:
                    errors.append((path, line_no, "table-wikilink-pipe", spec, "table alias pipe must be escaped as \\|"))
                if frontmatter_end is not None and line_no <= frontmatter_end and (line_no == 1 or frontmatter_open):
                    # Frontmatter is the initial delimited block only.
                    start = text.splitlines()
                    if line_no > 1 and all(x.strip() != "---" for x in start[1:line_no]):
                        line_text = line
                        quoted = re.search(r'"[^"\n]*\[\[[^\]]+\]\][^"\n]*"', line_text)
                        if not quoted:
                            errors.append((path, line_no, "frontmatter-wikilink-quoting", target, "frontmatter wikilink must be inside double quotes"))
            for match in MD_RE.finditer(line):
                target = match.group(2)
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                kind = "images" if match.group(0).startswith("!") else "markdown-links"
                counts[kind] += 1
                ok, reason = resolve_md(target, path)
                if not ok:
                    errors.append((path, line_no, kind[:-1] if kind.endswith("s") else kind, target, reason))

    def rel(path: Path) -> str:
        return path.relative_to(ROOT.parent).as_posix()

    print(f"Files scanned: {len(files)}")
    print("Links checked: " + ", ".join(f"{key}={counts[key]}" for key in ("wikilinks", "markdown-links", "images")))
    if pending:
        print(f"\nPending-generation targets ({len(pending)}):")
        for path, line, kind, target in pending:
            print(f"{rel(path)}:{line} — {kind} — {target} — target file not generated yet")
    print(f"\nBroken links ({len(errors)}):")
    for path, line, kind, target, reason in errors:
        print(f"{rel(path)}:{line} — {kind} — {target} — {reason}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
