# Vision-Language Analysis of Figures & Tables

*Covers `core/parsing/figure_analyzer.py`, the vision LLM factory, and how
this relates to `FigureTableAgent`'s caption-only analysis (see
`AGENTS_ARCHITECTURE.md`).*

## Two figure/table analysis paths exist — deliberately, not by accident

| | `FigureTableAgent` (text agent) | `figure_analyzer.py` (this doc) |
|---|---|---|
| Input | Caption text + table markdown | The actual cropped image pixels |
| Model | The regular text LLM | A local Ollama **vision** model |
| Always available | Yes | Only if a vision model is pulled |
| What it sees | What the caption *says* the figure shows | What the figure *actually* shows |

`FigureTableAgent`'s own docstring is explicit that it runs "caption-mode
only... NOT actual visual interpretation — that's the Vision-Optional
extension... which is unbuilt." This document covers the piece that closes
that exact gap — real image understanding, not just reading captions.

## Why a separate vision LLM factory (`get_vision_llm()`)

A vision-capable model is a different (larger, slower) model from the
text-reasoning model used everywhere else. `core/llm/llm_provider.py` keeps
`get_llm()` (text) and `get_vision_llm()` (image+text) as two separate
factories so:

- `settings.vision.enabled` can toggle the whole capability off without
  touching the text pipeline at all (default: **off**).
- Ollama's library renamed the Qwen vision family since the original
  settings were written (Qwen2-VL → Qwen2.5-VL) — a small model-tag
  translation map (`_OLLAMA_VISION_MODEL_MAP`) bridges the config's
  human-friendly names (`qwen2-vl-7b`) to the actual pullable Ollama tags
  (`qwen2.5vl:7b`), the same pattern already used for text models.

## Pipeline (`figure_analyzer.py::analyze_figures`)

```
ParsedPaper.figures (bbox + page + caption only, from Docling)
   │
   ▼  for each figure, up to settings.vision.max_figures_per_paper:
crop_figure()  -- PyMuPDF, converts Docling's bbox (bottom-left-origin,
   │              native PDF space) into PyMuPDF's rect (top-left-origin)
   ▼
base64-encode the cropped PNG
   │
   ▼
get_vision_llm().invoke([SystemMessage, HumanMessage(text + image_url)])
   │              -- multimodal message format confirmed against the
   │                 installed langchain_ollama source directly, not assumed
   ▼
Figure.image_path + Figure.ocr_text populated
   (ocr_text was a field already reserved in the schema for exactly this,
    unused until this work)
```

- **No-op by default.** If `settings.vision.enabled` is `False` (the
  default), `analyze_figures()` returns the `ParsedPaper` completely
  unchanged — the rest of the pipeline behaves identically whether or not a
  vision model is available.
- **Fails soft, per figure.** A crop failure or a single bad VLM call is
  logged and that one figure is left as-is — one bad crop never blocks the
  rest of the paper's review.
- **The prompt is deliberately factual, not speculative**: "describe
  factually what it shows... do not speculate about anything not visible in
  the image itself," and explicitly asks the model to say so plainly if the
  image is illegible rather than guess.

## The multimodal message format (a real gotcha, verified not assumed)

Rather than guessing the image-content-block shape, the installed
`langchain_ollama` package source was read directly to confirm it. Both a
bare-string and a nested-dict `image_url` format are accepted; this code
uses the simpler bare-string form:
```python
{"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
```

## Verified

- End-to-end wiring test with a **mocked** vision LLM: a real bbox-based crop
  from an actual sample PDF was produced on disk, the correct
  `SystemMessage`/`HumanMessage` structure was sent, and the mocked response
  correctly landed in `Figure.ocr_text`.
- **Empirically confirmed the vision/text split is real, not just a config
  flag**: pointed the vision client at the already-pulled text-only
  `qwen2.5:7b` model and got a hard `400` directly from Ollama —
  `"Multimodal data provided, but model does not support multimodal
  requests"` — proving there's no silent degraded-quality fallback; a real
  vision-tagged model is required.

## Known gap

**No vision model is currently pulled** (`ollama pull qwen2.5vl:7b` or the
smaller `:3b` variant) — this was an explicit, deliberate choice made this
session, not an oversight. The live pipeline dashboard correctly reports this
stage as "skipped" rather than faking a result. Pulling a model and setting
`VISION__ENABLED=true` is all that's needed to light this stage up for real.
