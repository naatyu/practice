# README learning roadmap

The root README is the final presentation. It embeds transparent SVGs directly:
there is no hosted website, interactive inspector, JavaScript, or external font.

The roadmap uses system typography, unframed colored lanes, and distinct
complete/in-progress/planned/deferred markers. Light and dark palettes are
selected with GitHub's supported picture markup. Compact two-column variants
keep station labels readable on narrow screens. Only lane headings, topic labels,
and per-topic statuses are visible; scope details stay in the written notes.
SVG stations do not pretend to be interactive.

Complete means complete for educational scope. Scope qualifiers, authorship
notes and detailed resources remain in roadmap.json and the topic READMEs;
ROADMAP.md retains the full plan. Tracks are thematic, not strict prerequisites.

## Updating

Edit roadmap.json and the detailed notes together, then regenerate:

```bash
uv run python docs/roadmap/build.py
uv run python docs/roadmap/build.py --check
```

The standard-library generator validates local resource paths, topic IDs,
statuses, label shape and the six-station overview limit. It generates:

- roadmap-light.svg and roadmap-dark.svg: desktop README views.
- roadmap-light-mobile.svg and roadmap-dark-mobile.svg: narrow-screen views.
- index.html: a local rendering check using the README's actual picture markup.

Generated SVGs are committed alongside their source, so GitHub visitors need no
build tools. The check detects stale outputs and missing README theme sources,
but cannot automatically verify agreement with narrative learning notes.
Do not edit generated assets directly. New lanes need explicit palette entries.

## Checking the appearance locally

Open index.html in a browser; it follows the browser's light/dark preference.
This is only a rendering aid, not a separate website or publishing target.
template.html contains that minimal local shell.

Optional browser checks and screenshots, using installed Chrome:

```bash
uv run --no-project --python .venv/bin/python --with playwright python docs/roadmap/preview.py
```

Checks cover the actual image-source choices, all four SVGs, transparency,
legible/unclipped labels and layouts from 320px to desktop. Screenshots are kept
in ignored previews/. Playwright is not a project dependency. The earlier
website-style draft and its obsolete screenshots are not part of the maintained assets.

This visualization and its tooling are AI-authored repository support, not a
learner-authored LLM algorithm.

Reference: [GitHub's theme-aware picture support](https://github.blog/changelog/2022-08-15-specify-theme-context-for-images-in-markdown-ga/).
