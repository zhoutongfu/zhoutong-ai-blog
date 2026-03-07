# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
quarto preview       # Live preview the site locally (hot-reloads on save)
quarto render        # Build the full site to docs/
```

## Architecture

This is a **Quarto blog** (`_quarto.yml` sets `type: website`, output to `docs/`).

- **`index.qmd`** — Homepage. Uses a Quarto listing (`id: research-logs`) that auto-discovers posts from the `posts/` directory, sorted by date descending.
- **`posts/<slug>/index.qmd`** — Each post is a self-contained directory. The YAML frontmatter (`title`, `date`, `categories`, `description`, `image`) drives what appears in the homepage listing.
- **`posts/<slug>/images/`** — Images for each post, referenced via relative paths in the `.qmd` file.
- **`_quarto.yml`** — Global site config: theme (cosmo), TOC settings, navbar links.
- **`styles.css`** — Custom CSS overrides.
- **`docs/`** — Auto-generated build output. Never edit directly.

## Adding a New Post

1. Create `posts/<new-slug>/index.qmd` with frontmatter:
   ```yaml
   ---
   title: "..."
   subtitle: "..."
   author: "Zhoutong"
   date: "YYYY-MM-DD"
   categories: [LLMs, ...]
   format:
     html:
       toc: true
       number-sections: true
   ---
   ```
2. Add images to `posts/<new-slug>/images/` and reference them with relative paths.
3. The post appears automatically on the homepage listing after `quarto render`.

## Post Style Conventions

- Technical depth: first-principles derivations with math (LaTeX via `$$...$$`), code blocks, and figures.
- Structure: Introduction → Background/Motivation → Core Mechanism → Taxonomy or Comparison → Conclusion → References.
- Figures use image + italicized caption directly below (not Quarto's caption system).
- Comparison tables use `| :--- |` left-aligned columns; wide tables use `::: {.column-page}` wrapper.
- References section at the end, numbered and bolded by paper title.
