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
    print(f"\n--- All {len(parsed.tables)} table(s) (markdown, truncated to 400 chars each) ---")
    for t in parsed.tables:
        print(f"\n[{t.table_id}] page {t.page}, caption: {t.caption!r}")
        print(t.markdown[:400])

if parsed.figures:
    print(f"\n--- Cropping all {len(parsed.figures)} figure(s) ---")
    for fig in parsed.figures:
        print(f"\n[{fig.figure_id}] page {fig.page}, bbox {fig.bbox}, caption: {fig.caption!r}")
        if fig.bbox and fig.page:
            out_path = f"./data/parsed_cache/{fig.figure_id}.png"
            try:
                crop_figure(pdf_path, fig.page, fig.bbox, out_path)
                print(f"  -> saved: {out_path}")
            except Exception as e:
                print(f"  -> crop failed: {e}")
        else:
            print("  -> no bbox/page available, skipping")
else:
    print("\nNo figures detected in this PDF.")