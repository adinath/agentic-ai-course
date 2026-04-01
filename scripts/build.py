#!/usr/bin/env python3
"""
Build script for Agentic AI Course.
Reads content/ markdown files + course.json, converts to HTML,
injects into src/template.html, writes dist/index.html.

No external dependencies — stdlib only.
"""

import json
import re
import html as html_mod
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONTENT_DIR = ROOT / "content"
SRC_DIR = ROOT / "src"
DIST_DIR = ROOT / "dist"

# ─────────────────────────────────────────
# Callout config: type → (icon, css-class)
# ─────────────────────────────────────────
CALLOUT_META = {
    "info":    ("📘", "info"),
    "tip":     ("💡", "tip"),
    "warning": ("⚠️", "warning"),
    "example": ("🔍", "example"),
}


# ════════════════════════════════════════
# YAML frontmatter parser (stdlib-only)
# ════════════════════════════════════════
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (meta_dict, body_without_frontmatter)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    meta = {}
    current_key = None
    current_list = None
    for line in fm_block.splitlines():
        if not line.strip():
            continue
        # List item under a key
        if line.startswith("  - ") and current_key and isinstance(current_list, list):
            val = line.strip()[2:].strip()
            # Check if it's a dict item (key: value)
            if ":" in val and not val.startswith('"'):
                # Could be nested dict list
                if isinstance(current_list[-1] if current_list else None, dict):
                    pass
                # Start a new dict in list
                k2, v2 = val.split(":", 1)
                if current_list and isinstance(current_list[-1], dict) and k2.strip() not in current_list[-1]:
                    current_list[-1][k2.strip()] = v2.strip().strip('"')
                else:
                    current_list.append({k2.strip(): v2.strip().strip('"')})
            else:
                current_list.append(val.strip().strip('"'))
            continue
        # Nested list continuation (e.g. topics sub-keys)
        if line.startswith("    ") and current_list and isinstance(current_list[-1], dict):
            stripped = line.strip()
            if ":" in stripped:
                k2, v2 = stripped.split(":", 1)
                current_list[-1][k2.strip()] = v2.strip().strip('"')
            continue
        # Top-level key: value
        if ":" in line and not line.startswith(" "):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"')
            if val == "":
                # Next lines are a list
                current_key = key
                current_list = []
                meta[key] = current_list
            else:
                meta[key] = val
                current_key = key
                current_list = None
    return meta, body


# ════════════════════════════════════════
# HTML escape helper
# ════════════════════════════════════════
def esc(s: str) -> str:
    return html_mod.escape(s)


# ════════════════════════════════════════
# Inline markdown → HTML
# ════════════════════════════════════════
def inline_md(text: str) -> str:
    """Convert inline markdown (bold, italic, code, <br/>) to HTML."""
    # Preserve existing HTML tags and entities — don't double-escape
    # Bold+italic: ***text***
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    # Bold: **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code: `text`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return text


# ════════════════════════════════════════
# Markdown → HTML converter
# ════════════════════════════════════════
def md_to_html(body: str) -> str:
    """
    Convert course markdown to HTML.
    Handles: headings, paragraphs, bullet lists, fenced code blocks,
    :::type callouts, :::lab lab blocks.
    """
    lines = body.split("\n")
    html_parts = []
    i = 0

    def flush_paragraph(buf):
        """Render a paragraph buffer as <p class="prose">."""
        text = " ".join(buf).strip()
        if text:
            html_parts.append(f'<p class="prose">{inline_md(text)}</p>')

    while i < len(lines):
        line = lines[i]

        # ── Fenced code block ──────────────────────────────
        if line.startswith("```"):
            lang = line[3:].strip() or "text"
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            # Remove leading/trailing blank lines from code
            while code_lines and not code_lines[0].strip():
                code_lines.pop(0)
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()
            code_content = esc("\n".join(code_lines))
            lang_display = lang.upper()
            html_parts.append(
                f'<div class="code-block">'
                f'<div class="code-header">'
                f'<div class="code-dots"><div class="code-dot"></div><div class="code-dot"></div><div class="code-dot"></div></div>'
                f'<span class="code-lang">{lang_display}</span>'
                f'<button class="code-copy" onclick="copyCode(this)">copy</button>'
                f'</div>'
                f'<div class="code-body"><pre>{code_content}</pre></div>'
                f'</div>'
            )
            i += 1
            continue

        # ── Callout / lab block :::type Title ─────────────
        m = re.match(r'^:::(\w+)\s*(.*)', line)
        if m:
            block_type = m.group(1).lower()
            block_title = m.group(2).strip()
            # Collect until closing :::
            content_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != ":::":
                content_lines.append(lines[i])
                i += 1

            if block_type == "lab":
                # Parse lab title (may contain "Lab X.Y — Title")
                # Extract lab number and title
                lab_match = re.match(r'Lab\s+([\d.]+)\s*[—\-–]\s*(.*)', block_title)
                if lab_match:
                    lab_num = lab_match.group(1)
                    lab_name = lab_match.group(2)
                else:
                    lab_num = ""
                    lab_name = block_title

                # Parse objectives as list items
                obj_html = ""
                obj_items = []
                current_buf = []
                for cl in content_lines:
                    cl_stripped = cl.strip()
                    if cl_stripped.startswith("- "):
                        if current_buf:
                            obj_items.append(" ".join(current_buf))
                            current_buf = []
                        obj_items.append(cl_stripped[2:].strip())
                    elif cl_stripped.startswith("**") or (cl_stripped and current_buf):
                        current_buf.append(cl_stripped)
                if current_buf:
                    obj_items.append(" ".join(current_buf))

                for item in obj_items:
                    obj_html += f'<li>{inline_md(item)}</li>\n'

                lab_label_html = f'Lab {lab_num}' if lab_num else 'Lab'
                html_parts.append(
                    f'<div class="lab-block">'
                    f'<div class="lab-header">'
                    f'<div class="lab-icon">🧪</div>'
                    f'<div><div class="lab-label">{lab_label_html}</div>'
                    f'<div class="lab-title">{esc(lab_name)}</div></div>'
                    f'</div>'
                    f'<div class="lab-body">'
                    f'<ul class="lab-objectives">{obj_html}</ul>'
                    f'</div>'
                    f'</div>'
                )
            else:
                # info / tip / warning / example
                icon, css_class = CALLOUT_META.get(block_type, ("ℹ️", block_type))
                # Render content lines as inline HTML (support <br/> and bold)
                content_html_parts = []
                for cl in content_lines:
                    cl_stripped = cl.strip()
                    if cl_stripped:
                        # Convert code fences inside callouts to <code>
                        content_html_parts.append(inline_md(cl_stripped))
                content_inner = "<br/>\n".join(content_html_parts)
                html_parts.append(
                    f'<div class="callout {css_class}">'
                    f'<div class="callout-bar"></div>'
                    f'<div class="callout-body">'
                    f'<div class="callout-title"><span class="icon">{icon}</span>{esc(block_title)}</div>'
                    f'<div class="callout-content">{content_inner}</div>'
                    f'</div>'
                    f'</div>'
                )
            i += 1
            continue

        # ── ATX headings ─────────────────────────────────────
        # h4: ####
        m = re.match(r'^####\s+(.*)', line)
        if m:
            text = m.group(1).strip()
            # Strip trailing {#id} anchors from heading display text
            text = re.sub(r'\s*\{#[\w-]+\}$', '', text)
            html_parts.append(f'<div class="subtopic-heading">{esc(text)}</div>')
            i += 1
            continue

        # h3: ###
        m = re.match(r'^###\s+(.*)', line)
        if m:
            text = m.group(1).strip()
            # Extract optional {#id} anchor
            anchor_match = re.search(r'\{#([\w-]+)\}', text)
            anchor_id = anchor_match.group(1) if anchor_match else None
            text = re.sub(r'\s*\{#[\w-]+\}', '', text).strip()
            # Strip "Topic: " prefix for topic-heading display
            topic_prefix = re.match(r'^Topic:\s*(.*)', text)
            if topic_prefix:
                display = topic_prefix.group(1)
                tag = f'<div class="topic" id="{anchor_id}">' if anchor_id else '<div class="topic">'
                html_parts.append(f'{tag}<div class="topic-heading">{esc(display)}</div>')
                # Note: topics are not closed here; they close at next topic/end
                # We use a simple approach: wrap everything until next ### or ##
            else:
                if anchor_id:
                    html_parts.append(f'<div class="subtopic" id="{anchor_id}"><div class="subtopic-heading">{esc(text)}</div>')
                else:
                    html_parts.append(f'<div class="subtopic"><div class="subtopic-heading">{esc(text)}</div>')
            i += 1
            continue

        # h2: ##
        m = re.match(r'^##\s+(.*)', line)
        if m:
            text = m.group(1).strip()
            anchor_match = re.search(r'\{#([\w-]+)\}', text)
            anchor_id = anchor_match.group(1) if anchor_match else None
            text = re.sub(r'\s*\{#[\w-]+\}', '', text).strip()
            if anchor_id:
                html_parts.append(f'<h2 id="{anchor_id}">{esc(text)}</h2>')
            else:
                html_parts.append(f'<h2>{esc(text)}</h2>')
            i += 1
            continue

        # ── Bullet list item ─────────────────────────────────
        if line.startswith("- ") or line.startswith("* "):
            # Collect all consecutive list items
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("* ")):
                item_text = lines[i][2:].strip()
                items.append(item_text)
                i += 1
            items_html = "".join(f"<li>{inline_md(item)}</li>" for item in items)
            html_parts.append(f'<ul class="bullet-list">{items_html}</ul>')
            continue

        # ── Empty line ───────────────────────────────────────
        if not line.strip():
            i += 1
            continue

        # ── Regular paragraph ────────────────────────────────
        # Collect consecutive non-empty, non-special lines
        para_lines = []
        while i < len(lines):
            l = lines[i]
            if (not l.strip() or l.startswith("#") or l.startswith("```")
                    or l.startswith(":::") or l.startswith("- ") or l.startswith("* ")):
                break
            para_lines.append(l.strip())
            i += 1
        if para_lines:
            text = " ".join(para_lines)
            html_parts.append(f'<p class="prose">{inline_md(text)}</p>')
        continue

    return "\n".join(html_parts)


# ════════════════════════════════════════
# Session HTML builder
# ════════════════════════════════════════
def build_session_html(session_meta: dict, md_file: Path, is_first_in_day: bool) -> str:
    """Build full session HTML from markdown file and session metadata."""
    raw = md_file.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(raw)

    sid = fm.get("id", session_meta["id"])
    number = fm.get("number", session_meta["number"])
    title = fm.get("title", session_meta["title"])
    time_str = fm.get("time", "")
    duration = fm.get("duration", "")
    sess_type = fm.get("type", "")

    # Split body: first paragraph(s) before first ### heading = session desc
    desc_lines = []
    rest_lines = []
    found_heading = False
    for line in body.splitlines():
        if not found_heading and (line.startswith("###") or line.startswith("##")):
            found_heading = True
        if found_heading:
            rest_lines.append(line)
        else:
            desc_lines.append(line)

    desc_text = " ".join(l.strip() for l in desc_lines if l.strip())
    rest_body = "\n".join(rest_lines)

    # Build session meta spans
    meta_spans = []
    if time_str:
        meta_spans.append(f'<span>{esc(time_str)}</span>')
    if duration:
        if meta_spans:
            meta_spans.append('<span class="session-meta-dot"></span>')
        meta_spans.append(f'<span>{esc(duration)}</span>')
    if sess_type:
        if meta_spans:
            meta_spans.append('<span class="session-meta-dot"></span>')
        meta_spans.append(f'<span>{esc(sess_type)}</span>')
    meta_html = "".join(meta_spans)

    desc_html = f'<p class="session-desc">{inline_md(desc_text)}</p>' if desc_text else ""

    # Convert rest of markdown body, then wrap topics properly
    body_html_raw = md_to_html(rest_body)
    body_html = wrap_topics(body_html_raw)

    parts = [f'<div class="session" id="{sid}">']
    parts.append(
        f'  <div class="session-header">'
        f'<div class="session-num">{esc(number)}</div>'
        f'<div class="session-info">'
        f'<div class="session-title">{esc(title)}</div>'
        f'<div class="session-meta">{meta_html}</div>'
        f'</div></div>'
    )
    if desc_html:
        parts.append(f'  {desc_html}')
    parts.append(body_html)
    parts.append('</div>')
    return "\n".join(parts)


def wrap_topics(html: str) -> str:
    """
    Post-process: ensure every <div class="topic"> is properly closed,
    and subtopics inside topics are closed before the next topic.
    This is a structural pass over the generated HTML.
    """
    # The md_to_html generates <div class="topic" id="..."> without closing tags
    # for topics. We need to close them before the next topic div or at end.
    # Simple approach: track open divs.

    # Split into lines and add closing tags where needed
    lines = html.split("\n")
    result = []
    in_topic = False
    in_subtopic = False

    for line in lines:
        stripped = line.strip()

        # Detect start of a new topic
        is_new_topic = bool(re.match(r'<div class="topic"', stripped))

        if is_new_topic:
            # Close any open subtopic and topic
            if in_subtopic:
                result.append('</div><!-- /subtopic -->')
                in_subtopic = False
            if in_topic:
                result.append('</div><!-- /topic -->')
            in_topic = True
            result.append(line)
            continue

        # Detect start of a subtopic
        is_new_subtopic = bool(re.match(r'<div class="subtopic"', stripped))
        if is_new_subtopic:
            if in_subtopic:
                result.append('</div><!-- /subtopic -->')
            in_subtopic = True
            result.append(line)
            continue

        result.append(line)

    # Close any remaining open divs
    if in_subtopic:
        result.append('</div><!-- /subtopic -->')
    if in_topic:
        result.append('</div><!-- /topic -->')

    return "\n".join(result)


# ════════════════════════════════════════
# Sidebar nav builder
# ════════════════════════════════════════
def build_sidebar_nav(days: list[dict]) -> str:
    parts = []
    for di, day in enumerate(days):
        day_num = day["number"]
        day_id = f"dg-{day_num}"
        ds_id = f"ds-{day_num}"
        # Day 1 starts open and active
        toggle_classes = "day-toggle"
        sessions_classes = "day-sessions"
        if day_num == 1:
            toggle_classes += " open active-day"
            sessions_classes += " open"

        # Short day label for sidebar
        short_labels = {1: "Foundations", 2: "Memory & Context", 3: "Multi-Agent & Evals"}
        short_label = short_labels.get(day_num, f"Day {day_num}")

        parts.append(f'    <!-- Day {day_num} -->')
        parts.append(f'    <div class="day-group" id="{day_id}">')
        parts.append(
            f'      <button class="{toggle_classes}" onclick="toggleDay({day_num})">'
            f'<span class="day-badge">{day_num}</span>'
            f'<span>{esc(short_label)}</span>'
            f'<span class="day-toggle-arrow">▶</span>'
            f'</button>'
        )
        parts.append(f'      <div class="{sessions_classes}" id="{ds_id}">')

        for si, sess in enumerate(day["sessions"]):
            sid = sess["id"]
            snum = sess["number"]
            stitle = sess["title"]
            # First session of day 1 starts active
            btn_class = "nav-session-btn"
            if day_num == 1 and si == 0:
                btn_class += " active"

            # Read topics from the markdown frontmatter
            topics = sess.get("_topics", [])

            parts.append(f'        <div class="nav-session">')
            parts.append(
                f'          <button class="{btn_class}" onclick="scrollToSession(\'{sid}\')">'
                f'<span class="nav-session-num">{esc(snum)}</span>'
                f'<span class="nav-session-title">{esc(stitle)}</span>'
                f'</button>'
            )
            if topics:
                parts.append(f'          <div class="nav-topics" id="topics-{sid[1:]}">')
                for topic in topics:
                    tid = topic.get("id", "")
                    ttitle = topic.get("title", "")
                    parts.append(
                        f'            <a href="javascript:void(0)" class="nav-topic-link" '
                        f'onclick="scrollToTopic(\'{tid}\')" data-topic="{tid}">{esc(ttitle)}</a>'
                    )
                parts.append(f'          </div>')
            parts.append(f'        </div>')

        parts.append(f'      </div>')
        parts.append(f'    </div>')
        parts.append('')

    return "\n".join(parts)


# ════════════════════════════════════════
# Hero section builder
# ════════════════════════════════════════
def build_hero(course: dict) -> str:
    hero = course.get("hero", {})
    eyebrow = hero.get("eyebrow", "3-Day Intensive Course")
    tagline = hero.get("tagline", "Agentic AI<br/>for <em>Developers</em>")
    description = hero.get("description", "")
    stats = hero.get("stats", [])

    stats_html = ""
    for stat in stats:
        stats_html += (
            f'<div class="hero-stat">'
            f'<span class="hero-stat-val">{esc(stat["value"])}</span>'
            f'<span class="hero-stat-label">{esc(stat["label"])}</span>'
            f'</div>'
        )

    return (
        f'  <!-- HERO -->\n'
        f'  <section class="hero fade-up">\n'
        f'    <div class="hero-grid"></div>\n'
        f'    <div class="hero-eyebrow">{esc(eyebrow)}</div>\n'
        f'    <h1 class="hero-title">{tagline}</h1>\n'
        f'    <p class="hero-subtitle">{esc(description)}</p>\n'
        f'    <div class="hero-stats">\n'
        f'      {stats_html}\n'
        f'    </div>\n'
        f'  </section>\n'
    )


# ════════════════════════════════════════
# Day section builder
# ════════════════════════════════════════
def build_day_section(day: dict, day_content_dir: Path) -> str:
    day_num = day["number"]
    day_id = day["id"]
    banner_color = day.get("banner_color", "#3b82f6")
    banner_label = day.get("banner_label", f"Day {day_num} · 3 Sessions")
    banner_title = day.get("banner_title", day["title"])
    banner_sub = day.get("banner_sub", "")

    parts = []
    parts.append(f'\n  <!-- Day {day_num} -->')
    parts.append(f'  <div class="day-section" id="{day_id}">')
    parts.append(f'    <div class="day-banner">')
    parts.append(f'      <div class="day-banner-accent" style="background:{banner_color}"></div>')
    parts.append(f'      <div class="day-banner-body">')
    parts.append(f'        <div class="day-banner-label">{banner_label}</div>')
    parts.append(f'        <div class="day-banner-title">{banner_title}</div>')
    parts.append(f'        <div class="day-banner-sub">{banner_sub}</div>')
    parts.append(f'      </div>')
    parts.append(f'    </div>')

    for si, sess in enumerate(day["sessions"]):
        md_file = day_content_dir / sess["file"]
        if not md_file.exists():
            print(f"  WARNING: {md_file} not found, skipping.")
            continue

        if si > 0:
            parts.append(f'\n    <div class="session-divider"></div>')

        session_html = build_session_html(sess, md_file, si == 0)
        # Indent session content
        indented = "\n".join("    " + l if l.strip() else l for l in session_html.split("\n"))
        parts.append(indented)

    parts.append(f'  </div><!-- /day{day_num} -->')
    return "\n".join(parts)


# ════════════════════════════════════════
# Load topics from markdown frontmatter
# ════════════════════════════════════════
def load_session_topics(sessions: list, day_dir: Path) -> list:
    """Read each session's markdown file and extract topic list from frontmatter."""
    enriched = []
    for sess in sessions:
        md_file = day_dir / sess["file"]
        if md_file.exists():
            raw = md_file.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(raw)
            topics = fm.get("topics", [])
            # Normalise topics to list of dicts
            normalised = []
            for t in topics:
                if isinstance(t, dict):
                    normalised.append(t)
                elif isinstance(t, str):
                    normalised.append({"id": t, "title": t})
            sess = dict(sess)
            sess["_topics"] = normalised
        enriched.append(sess)
    return enriched


# ════════════════════════════════════════
# Main build
# ════════════════════════════════════════
def build():
    DIST_DIR.mkdir(exist_ok=True)

    # 1. Load course metadata
    course = json.loads((CONTENT_DIR / "course.json").read_text())
    sidebar_cfg = course.get("sidebar", {})

    # 2. Load day metadata + enrich with topics
    days = []
    for day_num in [1, 2, 3]:
        day_dir = CONTENT_DIR / f"day-{day_num}"
        meta_file = day_dir / "meta.json"
        if not meta_file.exists():
            print(f"  WARNING: {meta_file} not found, skipping day {day_num}.")
            continue
        day_meta = json.loads(meta_file.read_text())
        day_meta["sessions"] = load_session_topics(day_meta["sessions"], day_dir)
        days.append(day_meta)

    # 3. Build sidebar nav HTML
    sidebar_nav_html = build_sidebar_nav(days)

    # 4. Build sidebar footer (tech pills)
    tech_pills = sidebar_cfg.get("tech_pills", [])
    footer_html = "\n".join(f'    <span class="tech-pill">{esc(p)}</span>' for p in tech_pills)

    # 5. Build main content HTML
    main_parts = []
    main_parts.append(build_hero(course))
    for day in days:
        day_dir = CONTENT_DIR / f"day-{day['number']}"
        main_parts.append(build_day_section(day, day_dir))
    main_parts.append('\n    <div style="height:80px"></div>')
    main_content_html = "\n".join(main_parts)

    # 6. Load template + assets
    template = (SRC_DIR / "template.html").read_text(encoding="utf-8")
    styles = (SRC_DIR / "styles.css").read_text(encoding="utf-8")
    scripts = (SRC_DIR / "scripts.js").read_text(encoding="utf-8")

    # 7. Inject everything into template
    output = template
    output = output.replace("{{STYLES}}", styles)
    output = output.replace("{{SCRIPTS}}", scripts)
    output = output.replace("{{SIDEBAR_LOGO}}", esc(sidebar_cfg.get("logo", "Agentic AI")))
    output = output.replace("{{SIDEBAR_TITLE}}", sidebar_cfg.get("title", "3-Day Intensive<br/>Course"))
    output = output.replace("{{SIDEBAR_META}}", esc(sidebar_cfg.get("meta", "")))
    output = output.replace("{{SIDEBAR_NAV}}", sidebar_nav_html)
    output = output.replace("{{SIDEBAR_FOOTER}}", footer_html)
    output = output.replace("{{MAIN_CONTENT}}", main_content_html)

    # 8. Write output
    out_file = DIST_DIR / "index.html"
    out_file.write_text(output, encoding="utf-8")
    size_kb = out_file.stat().st_size / 1024
    print(f"Built: {out_file}  ({size_kb:.1f} KB)")
    return out_file


if __name__ == "__main__":
    build()
