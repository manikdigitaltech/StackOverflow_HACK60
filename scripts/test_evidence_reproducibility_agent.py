"""
Step 7 test (agent #6 of 9): Evidence & Reproducibility Agent.

Only needs parsed_paper (no literature context) -- but unlike Methodology,
this is the first agent that needs TABLE data, since it cross-checks
prose claims against actual reported numbers.

Run with: python -m scripts.test_evidence_reproducibility_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.evidence_reproducibility_agent import EvidenceReproducibilityAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_evidence_reproducibility_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}")
print(f"Paper has {len(parsed.tables)} table(s) extracted.\n")

llm = get_llm()
prompt_manager = PromptManager()

print("Running Evidence & Reproducibility Agent (calls the LLM, may take a moment on CPU)...")
agent = EvidenceReproducibilityAgent(llm=llm, prompt_manager=prompt_manager)
result = agent.run({"parsed_paper": parsed})

print(f"\n--- Overall Rating: {result.overall_rating.upper()} ---")

print(f"\n--- Claim Verdicts ({len(result.claim_verdicts)} claims checked against tables) ---")
for c in result.claim_verdicts:
    print(f"  [{c.verdict.upper()}] {c.claim}")
    print(f"      {c.note}")

print(f"\n--- Reproducibility Verdicts ({len(result.reproducibility_verdicts)} of 5 expected) ---")
for v in result.reproducibility_verdicts:
    print(f"  [{v.assessment.upper()}] {v.aspect}")
    print(f"      {v.note}")

print("\n--- Reasoning ---")
print(result.reasoning)

print("\n--- Completeness check ---")
expected_aspects = {"code_availability", "hyperparameter_details", "dataset_availability",
                    "training_details", "compute_requirements"}
actual_aspects = {v.aspect for v in result.reproducibility_verdicts}
missing = expected_aspects - actual_aspects
if missing:
    print(f"WARNING: missing reproducibility verdicts for: {missing}")
else:
    print("All 5 expected reproducibility aspects got a verdict -- coverage is complete.")

if len(result.claim_verdicts) == 0:
    print("WARNING: no claims were checked at all -- the agent may have failed to "
          "extract any quantitative claims from the abstract/intro.")

print("\nSanity check: claim verdicts should cite ACTUAL numbers from the tables "
      "printed above (e.g. '62.9% in Table 1', '8.3x reduction') -- not vague "
      "restatements. For reproducibility, check whether it correctly notices "
      "things this paper's text genuinely never states (e.g. no base model "
      "name is given anywhere, no code repository is mentioned).")
