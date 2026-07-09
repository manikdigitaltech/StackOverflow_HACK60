"""
Reference Usage Agent test: checks how effectively a paper uses the
references already in its OWN bibliography -- the inverse of Citation
Agent, which checks whether external literature is missing from that same
bibliography. Deliberately skips Literature RAG (not needed here -- this
agent only reads the paper's own body + reference list).

Run with: python -m scripts.test_reference_usage_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.reference_usage_agent import ReferenceUsageAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_reference_usage_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}")
print(f"Paper has {len(parsed.references)} extracted references.\n")

llm = get_llm()
prompt_manager = PromptManager()

print("Running Reference Usage Agent (calls the LLM, may take a moment on CPU)...")
agent = ReferenceUsageAgent(llm=llm, prompt_manager=prompt_manager)
result = agent.run({"parsed_paper": parsed})

print(f"\n--- Overall Rating: {result.overall_rating.upper()} ---")

print(f"\n--- Reference Verdicts ({len(result.reference_verdicts)} of "
      f"{len(parsed.references)} extracted references checked) ---")
not_cited = []
for v in result.reference_verdicts:
    cited_label = "CITED" if v.cited_in_body else "NOT CITED"
    print(f'  [{cited_label}] [{v.role}] [usefulness={v.usefulness}] "{v.reference}"')
    print(f"      {v.evidence}")
    if not v.cited_in_body:
        not_cited.append(v.reference)

print("\n--- Summary ---")
print(result.summary)

print("\n--- Completeness check ---")
if len(result.reference_verdicts) < min(len(parsed.references), 60):
    print(f"NOTE: only {len(result.reference_verdicts)} verdict(s) produced -- some "
          f"references were skipped rather than every one being checked.")
else:
    print("Every reference (up to the 60-reference cap) got its own verdict -- checking is complete.")

print(f"\n{len(not_cited)} reference(s) judged not meaningfully cited: {not_cited if not_cited else '(none)'}")
print("\nWorth a manual check: pick one 'NOT CITED' entry and search the actual paper "
      "body yourself to confirm it's genuinely unused -- that judgment call is the one "
      "thing this script can't verify automatically.")
