"""
Step 7 test (last of the 9 agents to be built, though originally #2 in the
lineup): Figure & Table Agent.

Runs in caption-mode only (no vision) -- reasons from captions + table
data + a deterministic reference-count consistency check, not actual
image pixels.

Run with: python -m scripts.test_figure_table_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.figure_table_agent import FigureTableAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_figure_table_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}")
print(f"{len(parsed.figures)} figure(s), {len(parsed.tables)} table(s) extracted.\n")

llm = get_llm()
prompt_manager = PromptManager()

print("Running Figure & Table Agent (calls the LLM, may take a moment on CPU)...")
agent = FigureTableAgent(llm=llm, prompt_manager=prompt_manager)
result = agent.run({"parsed_paper": parsed})

print(f"\n--- Figure Summaries ({len(result.figure_summaries)}) ---")
for f in result.figure_summaries:
    print(f"  [{f.figure_id}] self-contained caption: {f.caption_self_contained}")
    print(f"      {f.interpretation}")

print(f"\n--- Table Summaries ({len(result.table_summaries)}) ---")
for t in result.table_summaries:
    print(f"  [{t.table_id}] self-contained caption: {t.caption_self_contained}")
    print(f"      {t.key_takeaway}")

print(f"\n--- Extraction Consistency Note (computed deterministically, not by the LLM) ---")
print(result.extraction_consistency_note)

print("\nSanity check: table key_takeaways should cite ACTUAL numbers from the "
      "tables (e.g. '62.9% avg score, best among compressed methods') -- not "
      "vague restatements. Figure interpretations should stick to what the "
      "CAPTION actually says, since this agent has no access to the real image.")
