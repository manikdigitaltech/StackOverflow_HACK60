"""
Step 3 test: parses one of your sample PDFs and prints a sanity-check summary.
Also test-crops the first detected figure (if any) to prove the PyMuPDF
coordinate conversion works, not just the Docling structure extraction.

Run with:  python -m scripts.test_parsing "./data/raw_papers/your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.parsing.figure_cropper import crop_figure

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_parsing "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]

print(f"Parsing: {pdf_path}")
parser = DoclingParser()
parsed = parser.parse(pdf_path)

print("\n--- ParsedPaper summary ---")
print(f"Title: {parsed.title}")
print(f"Abstract ({len(parsed.abstract)} chars): {parsed.abstract[:200]}{'...' if len(parsed.abstract) > 200 else ''}")
print(f"Sections ({len(parsed.sections)}): {[s.name for s in parsed.sections]}")
print(f"Tables: {len(parsed.tables)}")
print(f"Figures: {len(parsed.figures)}")
print(f"References parsed: {len(parsed.references)}")

if parsed.tables:
    print("\n--- First table (markdown, truncated) ---")
    print(parsed.tables[0].markdown[:500])

if parsed.figures:
    fig = parsed.figures[0]
    print(f"\n--- Test-cropping first figure (page {fig.page}, bbox {fig.bbox}) ---")
    if fig.bbox and fig.page:
        out_path = "./data/parsed_cache/test_figure_crop.png"
        try:
            crop_figure(pdf_path, fig.page, fig.bbox, out_path)
            print(f"Crop saved to: {out_path} — open it and confirm it actually shows the figure.")
        except Exception as e:
            print(f"Crop failed: {e}")
            print("This is the coordinate-conversion step mentioned in figure_cropper.py's docstring — "
                  "paste this error back and we'll fix it.")
    else:
        print("No bbox/page available for this figure — skipping crop test.")
else:
    print("\nNo figures detected in this PDF — try a different sample if you want to test figure cropping.")
