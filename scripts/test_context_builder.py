"""
Proves the section-priority context builder stays within budget regardless
of how long the source paper's raw text actually is.

Run with: python -m scripts.test_context_builder ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.parsing.context_builder import build_paper_context, build_reference_summary
from core.utils.token_budget import estimate_tokens

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_context_builder "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
parsed = DoclingParser().parse(pdf_path)

raw_full_text = "\n".join(s.text for s in parsed.sections)
raw_tokens = estimate_tokens(raw_full_text)
print(f"Raw section text (uncapped): ~{raw_tokens} tokens")

for budget in (2000, 6000):
    context = build_paper_context(parsed, max_tokens=budget)
    actual = estimate_tokens(context)
    print(f"\n--- build_paper_context(max_tokens={budget}) ---")
    print(f"Requested budget: {budget} | Actual estimated tokens: {actual} "
          f"| Within budget: {actual <= budget + 200}")  # small tolerance for title/header overhead

ref_summary = build_reference_summary(parsed)
print(f"\n--- Reference summary ---")
print(f"Estimated tokens: {estimate_tokens(ref_summary)}")
print(ref_summary[:300])
