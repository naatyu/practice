"""Generate transparent GitHub README roadmaps and a local rendering check."""

import argparse
import json
import re
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WIDTH = 880
STATUSES = {
    "complete": "Complete",
    "in-progress": "In progress",
    "planned": "Planned",
    "deferred": "Deferred",
}
THEMES = {
    "dark": {
        "ink": "#e6edf3",
        "muted": "#9da7b3",
        "check": "#0d1117",
        "lanes": {
            "model": "#79d9c9",
            "generation": "#8bb8f5",
            "attention-designs": "#baa7ed",
            "systems": "#e7b875",
            "adaptation": "#e59cac",
        },
    },
    "light": {
        "ink": "#1f2328",
        "muted": "#59636e",
        "check": "#ffffff",
        "lanes": {
            "model": "#087f72",
            "generation": "#245fa7",
            "attention-designs": "#7451ac",
            "systems": "#8d5909",
            "adaptation": "#a13c59",
        },
    },
}


def validate(data):
    ids = set()
    for lane in data["lanes"]:
        if lane["id"] not in THEMES["light"]["lanes"]:
            raise ValueError(f"Add a theme palette for lane {lane['id']}")
        if len(lane["topics"]) > 6:
            raise ValueError("Keep at most six overview stations per lane")
        for topic in lane["topics"]:
            if topic["id"] in ids or topic["status"] not in STATUSES:
                raise ValueError(f"Invalid or duplicate topic: {topic['id']}")
            ids.add(topic["id"])
            if not 1 <= len(topic["label"]) <= 2:
                raise ValueError("Use one or two short label lines per station")
            for _, target in topic["links"]:
                if not (ROOT / target.split("#")[0]).is_file():
                    raise ValueError(f"Missing local resource: {target}")


def text(x, y, content, cls, extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" {extra}>{escape(content)}</text>'


def marker(x, y, status, color, check, radius=6):
    # Hollow markers stay transparent. Route segments stop outside each marker.
    if status == "in-progress":
        r = radius + 2
        return (
            f'<path d="M{x} {y - r} l{r} {r} l-{r} {r} l-{r} -{r} Z" '
            f'fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
    if status == "deferred":
        return (
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="none" '
            f'stroke="{color}" stroke-width="1.5"/>'
            f'<path d="M{x - 2} {y - 3} v6 M{x + 2} {y - 3} v6" '
            f'stroke="{color}" stroke-width="1.3"/>'
        )
    fill = color if status == "complete" else "none"
    shape = (
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" '
        f'stroke="{color}" stroke-width="1.8"/>'
    )
    if status == "complete":
        shape += (
            f'<path d="M{x - 3} {y} l2 2 l4 -4" fill="none" stroke="{check}" '
            'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return shape


def make_svg(data, theme_name, compact=False):
    theme = THEMES[theme_name]
    width = 400 if compact else WIDTH
    columns = 2 if compact else 6
    top = 16
    height = top + sum(
        56 + ((len(lane["topics"]) + columns - 1) // columns) * 94
        for lane in data["lanes"]
    )
    description = "; ".join(
        lane["title"]
        + ": "
        + ", ".join(
            f"{topic['title']} ({STATUSES[topic['status']]})"
            for topic in lane["topics"]
        )
        for lane in data["lanes"]
    )
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="map-title map-desc">'
        ),
        '<title id="map-title">LLM implementation learning roadmap</title>',
        f'<desc id="map-desc">{escape(description)}. Tracks group topics, not strict prerequisites.</desc>',
        (
            "<style>"
            f'text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;fill:{theme["ink"]}}}'
            ".lane{font-size:19px;font-weight:600}"
            f".topic{{font-size:{16 if compact else 15}px;font-weight:500}}"
            f".status{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:10px;letter-spacing:.4px;fill:{theme['muted']}}}"
            "</style>"
        ),
    ]
    for lane in data["lanes"]:
        color = theme["lanes"][lane["id"]]
        parts.append(text(0, top + 4, lane["title"], "lane"))
        for slot, topic in enumerate(lane["topics"]):
            spacing = 198 if compact else 146
            x = 12 + (slot % columns) * spacing
            y = top + 40 + (slot // columns) * 94
            if slot < len(lane["topics"]) - 1 and slot % columns < columns - 1:
                complete = (
                    topic["status"] == lane["topics"][slot + 1]["status"] == "complete"
                )
                dash = "" if complete else ' stroke-dasharray="3 5"'
                opacity = ".65" if complete else ".35"
                parts.append(
                    f'<path d="M{x + 10} {y} H{x + spacing - 10}" fill="none" stroke="{color}" '
                    f'stroke-width="1.5" opacity="{opacity}"{dash}/>'
                )
            parts.append(
                f'<g data-topic="{topic["id"]}"><title>'
                f"{escape(topic['title'])}: {STATUSES[topic['status']]}</title>"
            )
            parts.append(marker(x, y, topic["status"], color, theme["check"]))
            for line, label in enumerate(topic["label"]):
                parts.append(text(x - 6, y + 27 + line * 19, label, "topic"))
            status_label = STATUSES[topic["status"]].upper()
            if topic.get("next"):
                status_label = (
                    "NEXT OPTION" if topic["status"] == "planned" else "THEORY DONE"
                )
            parts.append(text(x - 6, y + 68, status_label, "status"))
            parts.append("</g>")
        top += 56 + ((len(lane["topics"]) + columns - 1) // columns) * 94
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def preview_picture():
    readme = (ROOT / "README.md").read_text()
    block = readme.split("<!-- roadmap:start -->", 1)[1].split(
        "<!-- roadmap:end -->", 1
    )[0]
    picture = re.search(r"<picture>.*?</picture>", block, flags=re.DOTALL)
    if picture is None:
        raise ValueError(
            "The README must embed the themed roadmap in a picture element"
        )
    for asset in (
        "roadmap-light.svg",
        "roadmap-dark.svg",
        "roadmap-light-mobile.svg",
        "roadmap-dark-mobile.svg",
    ):
        if asset not in picture[0]:
            raise ValueError(f"Missing README theme source: {asset}")
    # Use the README's actual markup, changing only its relative asset paths.
    return picture[0].replace("docs/roadmap/", "")


def build(check=False):
    data = json.loads((HERE / "roadmap.json").read_text())
    validate(data)
    outputs = {
        f"roadmap-{name}{'-mobile' if compact else ''}.svg": make_svg(
            data, name, compact
        )
        for name in THEMES
        for compact in (False, True)
    }
    outputs["index.html"] = (
        (HERE / "template.html")
        .read_text()
        .replace("<!-- PICTURE -->", preview_picture())
    )
    stale = []
    for name, content in outputs.items():
        path = HERE / name
        if check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            path.write_text(content)
    if stale:
        raise SystemExit("Regenerate the roadmap: " + ", ".join(stale))
    print(
        "Roadmap outputs are current."
        if check
        else "Generated both SVG themes and local README preview."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if generated outputs are stale"
    )
    build(parser.parse_args().check)
