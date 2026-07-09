# Scientific Document Understanding — Parsing Pipeline

*Covers `core/parsing/`. Verified against the actual code and a real sample-PDF run.*

## Pipeline shape

```
PDF --> DoclingParser.parse()
          ├─ fitz (PyMuPDF): cheap page-count safety check
          ├─ Docling DocumentConverter: layout analysis + TableFormer + OCR-if-needed
          ├─ section_segmenter.segment_sections(): flat text stream -> named Sections + title + abstract
          ├─ reference_extractor.extract_references(): References section -> structured Reference list
          └─ Table/Figure extraction (bbox + caption only, no pixels)
                --> ParsedPaper (the one schema every downstream agent consumes)

figure_cropper.crop_figure() -- separate, on-demand step (see VLM doc):
  PDF + bbox --> cropped PNG, only when vision analysis actually runs
```

## `DoclingParser` (`docling_parser.py`)

- **Page-count safety valve.** Before running Docling's (expensive) full
  pipeline, a cheap PyMuPDF page count check runs first. If a PDF exceeds
  `settings.parsing.max_pages_hard_cap` (default 60), only the first N pages
  are processed and a warning is printed — protects against an accidental
  300-page dissertation upload hanging a review for minutes on CPU.
- **Docling does layout analysis, TableFormer (table structure recognition),
  and OCR only when needed** — this is real, general PDF understanding, not
  a text-only extractor.
- **Deliberately does NOT extract figure pixels.** Figures get bbox + page +
  caption only here — pixel extraction is a separate, on-demand concern
  (`figure_cropper.py`), so this module stays focused on document structure.
- **Table export** goes through `export_to_dataframe().to_markdown()`, with a
  fallback to `export_to_html()` and then a plain "(unavailable)" string if
  the installed Docling version's API differs — tables never silently
  disappear from a `ParsedPaper` due to a version mismatch.
- **Caption resolution** is non-trivial: Docling stores a figure/table's
  caption as a *reference* to another `TextItem`, not inline text. A small
  resolver walks that reference and stitches multi-part captions together.

## `section_segmenter.py` — turning a flat stream into structure

Docling hands back a flat, reading-order stream of `(label, text, page_no)`
tuples — it does not hand you "the abstract" as a field. This module is the
glue:

- **Title detection** happens *before* section segmentation, specifically so
  the title (wherever Docling actually labels it, or — if no PDF-level
  `title` label exists, which many PDFs don't have — falling back to the
  first non-empty text block) can be filtered out everywhere it recurs later
  (e.g. as a running header repeated on every page).
- **Canonicalization**: raw headings ("3.2 Methodology", "Related Work and
  Prior Approaches") are numbering-stripped and keyword-matched against a
  fixed vocabulary down to one of ~13 canonical names (`Abstract`, `Method`,
  `Experiments`, `Results`, `Related Work`, `Background`, `Discussion`,
  `Conclusion`, `Limitations`, `References`, `Appendix`, `Acknowledgements`,
  or a raw fallback). This canonical vocabulary is what `context_builder.py`'s
  priority weights and the agents' section-name checks both key off of.
- **Un-headed text** at the very start of a paper (before any detected
  heading — title-page boilerplate, venue notices) is bucketed as
  `"Preamble"` rather than dropped or mis-attributed.

## `reference_extractor.py` — structured, not prose

Deliberately shallow: extracts raw reference-entry text + a best-effort
publication year (regex for a 4-digit `19xx`/`20xx`), **not** full citation
metadata (author lists, venues). It's just enough for the Citation Agent to
check presence/absence — full bibliographic parsing wasn't needed for that job.

Splits on common numbered-list markers (`[1]`, `1.`, `1)`); falls back to
one-entry-per-line if no numbering pattern is detected, so an unusually
formatted reference list still yields *something* rather than one giant blob.

## `context_builder.py` — fitting any paper into a fixed token budget

The problem this solves: an LLM prompt has a fixed context window, but papers
range from a 9-page conference paper to a 60-page thesis chapter. Not every
section deserves equal space in that budget.

**Weighted-by-relevance truncation**, not naive truncation:

| Section | Weight | Section | Weight |
|---|---|---|---|
| Abstract, Method, Experiments, Results | 1.0 (Abstract always included in full, separately) | Discussion, Limitations | 0.4 |
| Ablation Study | 0.8 | Introduction | 0.35 |
| Related Work, Background | 0.25 | Appendix | 0.15 |
| Conclusion | 0.25 | Future Work | 0.15 |
| Preamble | 0.05 | Acknowledgements, References | 0.0 (excluded from prose entirely) |

Each section's token allocation is `remaining_budget × (its weight / total
weight of all included sections)` — so, e.g., Method/Experiments/Results
always get the lion's share regardless of how long Related Work happens to
run in a given paper. Sections truncated below 20 tokens are dropped
entirely rather than included as a useless sliver.

**References are excluded from the prose context on purpose** — the Citation
Agent checks claims against the structured `Reference` list directly (via
`build_reference_summary()`, a separate structured — not narrative — summary
capped at 60 entries), so dumping raw bibliography prose into every agent's
prompt would just waste token budget none of them need.

## Verified

Real end-to-end run against a sample PDF this session: correctly extracted a
real title, 10 canonical sections, 2 tables, 2 figures, and structured
references from a genuine paper (`DepthWeave-KV: Token-Adaptive Cross-Layer
Residual Factorization...`), through the live pipeline dashboard.
