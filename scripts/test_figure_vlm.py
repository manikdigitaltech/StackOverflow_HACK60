"""
Vision test: parses one of your sample PDFs, then runs figure_analyzer over it
to crop each figure and get a local VLM description -- proves the whole
Docling -> crop -> Ollama vision chain works end to end, not just the crop half
(that's already covered by scripts.test_parsing).

Requires a vision-capable model pulled in Ollama first, e.g.:
    ollama pull qwen2.5vl:7b        (maps from settings VISION__PROVIDER=qwen2-vl-7b)
    ollama pull llama3.2-vision:11b (maps from settings VISION__PROVIDER=llama3.2-vision-11b)

Run with:  python -m scripts.test_figure_vlm "./data/raw_papers/your_sample.pdf"
"""

import sys

from core.config.settings import settings
from core.parsing.docling_parser import DoclingParser
from core.parsing.figure_analyzer import analyze_figures

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_figure_vlm "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]

if not settings.vision.enabled:
    print("VISION__ENABLED is false in your .env -- forcing it on for this test run only.")
    settings.vision.enabled = True

print(f"Vision provider: {settings.vision.provider} @ {settings.vision.base_url}")
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Found {len(parsed.figures)} figure(s).")

print("\n--- Running figure_analyzer (crop + VLM describe) ---")
result = analyze_figures(parsed)

for fig in result.figures:
    print(f"\n[{fig.figure_id}] page {fig.page}, caption: {fig.caption!r}")
    if fig.image_path:
        print(f"  crop: {fig.image_path}")
        print(f"  VLM description: {fig.ocr_text}")
    else:
        print("  -> not analyzed (no bbox/page, or over max_figures_per_paper cap)")

print("\nIf every figure with a bbox got a plausible-sounding description above "
      "(not an error message), the vision pipeline is working.")
