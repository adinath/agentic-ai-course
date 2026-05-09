#!/usr/bin/env python3
"""
Splits dist/presentation.html into per-day standalone presentations:

  dist/day1_presentation.html
  dist/day2_presentation.html
  dist/day3_presentation.html

Each output preserves the original styling, navigation chrome, and
scripts. Only the slide deck content differs: each file contains the
day-break slide plus that day's content slides.

Re-run this script whenever dist/presentation.html is regenerated.

No external dependencies - stdlib only.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIST_DIR = ROOT / "dist"
SOURCE_FILE = DIST_DIR / "presentation.html"

DECK_OPEN_MARKER = '<div id="deck">'
DECK_CLOSE_MARKER = '</div><!-- /deck -->'

SLIDE_PATTERN = re.compile(
    r'^<div class="slide [^"]*" data-day="(?P<day>\d+)">.*?^</div>\n',
    re.DOTALL | re.MULTILINE,
)

DAY_TITLES = {
    1: "Day 1 - LLM Based Agents",
    2: "Day 2 - Single Agent Mastery",
    3: "Day 3 - Multi-Agent Systems",
}


def _split_shell(html: str) -> tuple[str, str, str]:
    """Return (header, deck_inner, footer) where:
      header   = everything from <!DOCTYPE> through `<div id="deck">\n`
      deck_inner = original slide content
      footer   = `</div><!-- /deck -->` through `</html>`
    """
    deck_open_idx = html.index(DECK_OPEN_MARKER)
    deck_close_idx = html.index(DECK_CLOSE_MARKER)

    header_end = deck_open_idx + len(DECK_OPEN_MARKER) + 1
    header = html[:header_end]
    deck_inner = html[header_end:deck_close_idx]
    footer = html[deck_close_idx:]
    return header, deck_inner, footer


def _collect_slides_by_day(deck_inner: str) -> dict[int, list[str]]:
    """Return {day_number: [raw_slide_block, ...]} preserving source order."""
    slides_by_day: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for match in SLIDE_PATTERN.finditer(deck_inner):
        day = int(match.group("day"))
        if day == 0:
            continue
        slides_by_day.setdefault(day, []).append(match.group(0))
    return slides_by_day


def _retitle(header: str, day_title: str) -> str:
    """Replace the original <title> with a day-specific title."""
    return re.sub(
        r'<title>[^<]*</title>',
        f'<title>{day_title}</title>',
        header,
        count=1,
    )


def _build_day_html(header: str, footer: str, day_slides: list[str], day_title: str) -> str:
    deck_body = "\n".join(day_slides).rstrip() + "\n\n"
    return _retitle(header, day_title) + "\n" + deck_body + footer


def split() -> list[Path]:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing source: {SOURCE_FILE}")

    html = SOURCE_FILE.read_text(encoding="utf-8")
    header, deck_inner, footer = _split_shell(html)
    slides_by_day = _collect_slides_by_day(deck_inner)

    written: list[Path] = []
    for day_num, slides in sorted(slides_by_day.items()):
        if not slides:
            print(f"  SKIP day {day_num}: no slides found")
            continue
        out_file = DIST_DIR / f"day{day_num}_presentation.html"
        day_html = _build_day_html(header, footer, slides, DAY_TITLES[day_num])
        out_file.write_text(day_html, encoding="utf-8")
        size_kb = out_file.stat().st_size / 1024
        print(f"  Wrote {out_file.name}  ({len(slides)} slides, {size_kb:.1f} KB)")
        written.append(out_file)
    return written


if __name__ == "__main__":
    split()
