"""
Step 7 test (agent #1 of 9): Paper Understanding Agent.

This proves the whole agent pattern works end-to-end -- prompt rendering,
token-budgeted context, Ollama JSON-mode enforcement, and Pydantic
validation -- before we replicate this pattern for the remaining 8 agents.

Run with: python -m scripts.test_paper_understanding_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.paper_understanding_agent import PaperUnderstandingAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_paper_understanding_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]

print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}\n")

print("Running Paper Understanding Agent (this calls the LLM, may take a moment on CPU)...")
llm = get_llm()  # json_mode=True by default now
prompt_manager = PromptManager()
agent = PaperUnderstandingAgent(llm=llm, prompt_manager=prompt_manager)

result = agent.run({"parsed_paper": parsed})

print("\n--- Summary ---")
print(result.summary)

print("\n--- Stated Contributions ---")
for i, c in enumerate(result.stated_contributions, 1):
    print(f"  {i}. {c}")

print("\n--- Key Terms ---")
print(", ".join(result.key_terms))

print("\nSanity check: the summary should accurately describe THIS paper "
      "(not a generic/hallucinated one), contributions should match what "
      "the paper's intro/abstract actually claims, and key terms should be "
      "genuinely central concepts, not generic ML boilerplate words.")
