"""
Diagnostic (not an agent): checks whether specific technical details --
like the paper's rank-value hyperparameters -- actually survive (a) raw
Docling text extraction and (b) the token-budget truncation in
build_paper_context(), before assuming an agent's miss is just a model
reasoning limitation.

Run with: python -m scripts.debug_method_section ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.parsing.context_builder import build_paper_context
from core.config.settings import settings

if len(sys.argv) < 2:
    print('Usage: python -m scripts.debug_method_section "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
parsed = DoclingParser().parse(pdf_path)

method_section = next((s for s in parsed.sections if s.name == "Method"), None)
if method_section is None:
    print("No 'Method' section found. Available sections:", [s.name for s in parsed.sections])
    sys.exit(1)

needles = ["ρ", "rho", "rank level", "residual rank", "= 0 or", "= 4", "= 8"]

print(f"--- RAW Method section text length: {len(method_section.text)} chars ---")
print("\n--- Presence check in RAW (post-extraction, pre-truncation) text ---")
for n in needles:
    print(f"  {n!r} found: {n in method_section.text}")

print(f"\n--- Full raw Method section text (for visual inspection) ---")
print(method_section.text)

context = build_paper_context(parsed, max_tokens=settings.parsing.prompt_token_budget)
print(f"\n\n--- Presence check in TRUNCATED context actually sent to agents ---")
for n in needles:
    print(f"  {n!r} found: {n in context}")

print("\nHow to read this:")
print("- If a needle is MISSING even in the RAW text -> Docling/PDF text extraction")
print("  mangled or dropped it (likely the 'ρ' math symbol specifically) -- a real")
print("  parsing limitation, not a truncation issue.")
print("- If present in RAW but MISSING in the truncated context -> our token-budget")
print("  truncation is cutting out this specific sentence -- a real, fixable issue")
print("  in context_builder.py's section truncation strategy.")
print("- If present in BOTH -> the text was available to the LLM the whole time,")
print("  and this is a genuine model reasoning/attention limitation, not a bug.")
