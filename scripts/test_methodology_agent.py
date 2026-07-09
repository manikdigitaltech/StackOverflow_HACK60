"""
Step 7 test (agent #4 of 9): Methodology Agent.

Only needs parsed_paper (no literature context, unlike Novelty) -- reasons
entirely from the paper's own text.

Run with: python -m scripts.test_methodology_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.methodology_agent import MethodologyAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_methodology_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}\n")

llm = get_llm()
prompt_manager = PromptManager()

print("Running Methodology Agent (calls the LLM, may take a moment on CPU)...")
agent = MethodologyAgent(llm=llm, prompt_manager=prompt_manager)
result = agent.run({"parsed_paper": parsed})

print(f"\n--- Soundness Rating: {result.soundness_rating.upper()} ---")

print(f"\n--- Aspect Verdicts ({len(result.aspect_verdicts)} of 5 expected) ---")
for v in result.aspect_verdicts:
    print(f"  [{v.assessment.upper()}] {v.aspect}")
    print(f"      {v.note}")

print("\n--- Missing Baselines ---")
if result.missing_baselines:
    for i, b in enumerate(result.missing_baselines, 1):
        print(f"  {i}. {b}")
else:
    print("  (none noted)")

print("\n--- Reasoning ---")
print(result.reasoning)

print("\n--- Completeness check ---")
expected_aspects = {"baseline_comparisons", "ablation_studies", "hyperparameter_justification",
                    "experimental_setup_clarity", "statistical_rigor"}
actual_aspects = {v.aspect for v in result.aspect_verdicts}
missing = expected_aspects - actual_aspects
if missing:
    print(f"WARNING: missing verdicts for: {missing}")
else:
    print("All 5 expected aspects got a verdict -- coverage is complete.")

print("\nSanity check: each verdict's 'note' should cite something SPECIFIC from "
      "this paper (a named baseline like 'StreamingLLM', a specific ablation "
      "result like the 62.9%->61.1% score drop, a specific rank value like "
      "rho=0/2/4/8) -- not a generic, could-apply-to-any-paper judgment.")
