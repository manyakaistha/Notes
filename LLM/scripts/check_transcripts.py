#!/usr/bin/env python3
"""Audit weekly ebook transcripts against their assigned PDF page spans."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEEKS = {1: (4, 20), 2: (21, 33), 3: (34, 44), 4: (45, 59)}
PAGE_MARKER = re.compile(r"^<!-- PDF p\.(\d+)\b[^>]*-->\s*$", re.MULTILINE)
WORD = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
BOX_LABELS = (
    "Definition",
    "Intuition",
    "Exam Tip",
    "Formula Insight",
    "Common Mistake",
    "Research Insight",
)


def main() -> int:
    print("Transcript audit (PDF page markers are the page boundaries)")
    print("Counts: $, [illegible, [possible-error, [Figure:, boxed labels")
    errors = 0
    for week, (first, last) in WEEKS.items():
        path = ROOT / "sources" / "ebook" / f"week-{week}-transcript.md"
        text = path.read_text(encoding="utf-8")
        markers = list(PAGE_MARKER.finditer(text))
        pages = [int(match.group(1)) for match in markers]
        expected = list(range(first, last + 1))
        counts = Counter(pages)
        missing = [page for page in expected if counts[page] == 0]
        duplicate = [page for page in expected if counts[page] > 1]
        in_range = [page for page in pages if first <= page <= last]
        order_ok = in_range == expected
        out_of_range = [page for page in pages if not first <= page <= last]
        incomplete_lines = [
            i for i, line in enumerate(text.splitlines(), 1) if "incomplete" in line.lower()
        ]

        page_text: dict[int, str] = {}
        for index, match in enumerate(markers):
            page = int(match.group(1))
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            if page in page_text:
                page_text[page] += "\n" + text[match.end() : end]
            else:
                page_text[page] = text[match.end() : end]

        total = {"lines": 0, "words": 0}
        marker_totals = Counter()
        short_pages = []
        for page in expected:
            body = page_text.get(page, "")
            words = len(WORD.findall(body))
            total["lines"] += len(body.splitlines())
            total["words"] += words
            stats = {
                "$": body.count("$"),
                "[illegible": body.count("[illegible"),
                "[possible-error": body.count("[possible-error"),
                "[Figure:": body.count("[Figure:"),
                "boxed": sum(
                    len(
                        re.findall(
                            rf"^\s*>\s*\*\*{re.escape(label)}\*\*",
                            body,
                            re.MULTILINE,
                        )
                    )
                    for label in BOX_LABELS
                ),
            }
            marker_totals.update(stats)
            if words < 40:
                short_pages.append((page, words))

        status_errors = []
        if missing:
            status_errors.append(f"missing pages {missing}")
        if duplicate:
            status_errors.append(f"duplicate pages {duplicate}")
        if not order_ok:
            status_errors.append(f"out-of-order markers (found {in_range})")
        if out_of_range:
            status_errors.append(f"out-of-range pages {out_of_range}")
        if incomplete_lines:
            status_errors.append(f"'incomplete' on lines {incomplete_lines}")
        errors += len(status_errors)

        print(f"\nW{week}: {path.name}")
        print(f"  pages: {len(in_range)}/{len(expected)} markers; expected {first}-{last}; order={'OK' if order_ok else 'FAIL'}")
        print(f"  transcript: {total['lines']} lines, {total['words']} words")
        print("  markers: " + ", ".join(f"{key}={marker_totals[key]}" for key in ("$", "[illegible", "[possible-error", "[Figure:", "boxed")))
        print(f"  pages under 40 words: {', '.join(f'p.{p} ({n})' for p, n in short_pages) if short_pages else 'none'}")
        print(f"  failures: {'; '.join(status_errors) if status_errors else 'none'}")

    print(f"\nAudit result: {'FAIL' if errors else 'PASS'} ({errors} structural failure(s); short pages are flagged above)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
