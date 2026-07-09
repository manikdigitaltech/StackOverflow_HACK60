"""
run_novelty_agent.py

CLI entry point: index a corpus and evaluate every paper in it (or a
single paper) end to end.

Usage:
    python -m scripts.run_novelty_agent --corpus data/novelty_corpus --output output_results
    python -m scripts.run_novelty_agent --corpus data/novelty_corpus --paper data/novelty_corpus/326.json --output output_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.agents.novelty import NoveltyEvaluationAgent, NoveltyEvaluationAgentError
from core.agents.novelty.config import DEFAULT_PATHS, TOP_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Novelty Evaluation Agent")
    parser.add_argument("--corpus", default=str(DEFAULT_PATHS.corpus_dir), help="Directory of paper JSON files")
    parser.add_argument("--output", default="output_results", help="Directory to write novelty reports")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of similar papers to retrieve")
    parser.add_argument("--paper", default=None, help="Evaluate a single external paper JSON file instead of the whole corpus")
    args = parser.parse_args()

    agent = NoveltyEvaluationAgent(top_k=args.top_k)
    agent.index_corpus(args.corpus)

    output_dir = Path(args.output)

    if args.paper:
        paper_path = Path(args.paper)
        paper_json = json.loads(paper_path.read_text(encoding="utf-8"))
        try:
            report = agent.evaluate(paper_json, paper_id=paper_path.stem)
        except NoveltyEvaluationAgentError as exc:
            print(f"Failed to evaluate '{paper_path}': {exc}")
            return
        agent.save_report(report, output_dir / f"{report.paper_id}_novelty.json")
        print(json.dumps(report.to_dict(), indent=2))
        return

    reports = agent.evaluate_all_indexed()
    for report in reports:
        agent.save_report(report, output_dir / f"{report.paper_id}_novelty.json")

    print(f"Scored {len(reports)} papers. Reports written to {output_dir}")
    for report in sorted(reports, key=lambda r: r.novelty_score):
        print(
            f"  {report.paper_id}: novelty={report.novelty_score:6.2f}  "
            f"confidence={report.confidence:6.2f}  recommendation={report.recommendation}"
        )


if __name__ == "__main__":
    main()
