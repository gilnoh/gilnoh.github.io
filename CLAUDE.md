# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal blog for Dr. Tae-Gil Noh (Gil Noh), hosted on GitHub Pages at https://gilnoh.github.io. The site uses the Minima theme and focuses on NLP, AI, and machine learning topics.

## Common Commands

```bash
# Local development (requires Jekyll installed)
bundle exec jekyll serve

# Build site
bundle exec jekyll build

# Install dependencies
bundle install
```

## Architecture

- **Theme**: Minima (Jekyll default theme, no custom layouts/includes)
- **Plugins**: jekyll-seo-tag, jekyll-sitemap
- **Content**: Blog posts in `_posts/` following Jekyll naming convention (`YYYY-MM-DD-title.md`)
- **Assets**: Images stored in `assets/images/`, with post-specific images in subdirectories matching post names

## Writing Posts

Posts use standard Jekyll front matter:
```yaml
---
layout: post
title: "Post Title"
date: YYYY-MM-DD
---
```

Images are referenced as `/assets/images/filename.png`. For posts with multiple images, create a subdirectory matching the post filename (e.g., `assets/images/2025-11-08-LLM-randomness-part1/`).
