"""Check the static README diagrams in both themes; optional Playwright/Chrome."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Installed transiently by the documented `uv run --with playwright` command.
from playwright.sync_api import sync_playwright  # ty: ignore[unresolved-import]

HERE = Path(__file__).resolve().parent
NS = {"svg": "http://www.w3.org/2000/svg"}


def main():
    output = HERE / "previews"
    output.mkdir(exist_ok=True)
    data = json.loads((HERE / "roadmap.json").read_text())
    expected_topics = sum(len(lane["topics"]) for lane in data["lanes"])
    for svg in HERE.glob("roadmap-*.svg"):
        root = ET.parse(svg).getroot()
        assert not root.findall(".//svg:rect", NS), "No panel backgrounds"
        assert not root.findall(".//svg:script", NS), "SVGs must stay static"
        assert len(root.findall(".//svg:g[@data-topic]", NS)) == expected_topics

    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome") or None, headless=True
        )
        page = browser.new_page(viewport={"width": 1060, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        for theme in ("light", "dark"):
            page.emulate_media(color_scheme=theme)
            for width in (320, 390, 768, 1060):
                page.set_viewport_size({"width": width, "height": 1100})
                page.goto((HERE / "index.html").as_uri())
                page.locator("img").wait_for()
                page.wait_for_function(
                    "document.querySelector('img').complete && document.querySelector('img').naturalWidth > 0"
                )
                suffix = "-mobile" if width < 768 else ""
                actual = page.locator("img").evaluate("(img) => img.currentSrc")
                assert actual.endswith(f"roadmap-{theme}{suffix}.svg"), actual
                assert not page.evaluate(
                    "document.documentElement.scrollWidth > innerWidth"
                ), f"Overflow at {width}px in {theme}"
                if width in (390, 1060):
                    name = f"readme-{theme}{'-mobile' if width == 390 else ''}.png"
                    page.screenshot(path=str(output / name), full_page=True)

            # Inspect the SVG at native resolution, not only the scaled image.
            for suffix in ("", "-mobile"):
                page.goto((HERE / f"roadmap-{theme}{suffix}.svg").as_uri())
                clipped = page.evaluate("""() => {
                    const box = document.querySelector("svg").viewBox.baseVal;
                    return [...document.querySelectorAll("text")].filter(text => {
                        const b = text.getBBox();
                        return b.x < 0 || b.y < 0 || b.x+b.width > box.width || b.y+b.height > box.height;
                    }).map(text => text.textContent);
                }""")
                assert not clipped, clipped
        assert not errors, errors
        browser.close()
    print(
        "Both README themes and compact variants pass; previews are in docs/roadmap/previews/."
    )


if __name__ == "__main__":
    main()
