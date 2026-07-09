"""
Runs the PeerRead ICLR-2017 evaluation harness -- the graded core of this
project (docs/CONTEXT.md item 3). The full 9-agent LangGraph review runs
against every test-split paper's real PDF, scored against PeerRead's own
accept/reject ground truth.

Writes one JSON line per paper to --output as soon as it completes (so a
crash partway through doesn't lose finished work) and skips any paper
already present in --output on a re-run, so this script is safe to resume.

Usage:
    python -m scripts.run_peerread_evaluation \\
        --peerread-dir data/peerread_raw --venue iclr_2017 \\
        --output output_results/peerread_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core.eval.peerread_harness import PaperResult, compute_metrics, load_test_set, run_single_paper
from core.graph.build_graph import build_review_graph
from core.parsing.docling_parser import DoclingParser

_RESULT_FIELDS = [
    "paper_id", "title", "ground_truth_accepted", "predicted_accept",
    "final_recommendation", "confidence", "elapsed_s", "error",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peerread-dir", type=Path, default=Path("data/peerread_raw"))
    parser.add_argument("--venue", default="iclr_2017")
    parser.add_argument("--output", type=Path, default=Path("output_results/peerread_eval.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N papers (smoke testing)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    results: list[PaperResult] = []
    if args.output.exists():
        errored_prior = 0
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            # Only a genuinely completed paper counts as "done" -- an errored
            # entry (e.g. Ollama was down) must be retried on resume, not
            # skipped forever just because a row for it already exists.
            if d.get("error") is None:
                already_done.add(d["paper_id"])
                results.append(PaperResult(**{k: d[k] for k in _RESULT_FIELDS}))
            else:
                errored_prior += 1
        logging.info(
            "Resuming: %d paper(s) already scored, %d previously-errored paper(s) will be retried",
            len(already_done), errored_prior,
        )
        # Rewrite clean -- drop stale errored rows now so a retried paper's
        # new outcome doesn't sit alongside a duplicate old error row.
        if errored_prior:
            args.output.write_text(
                "".join(json.dumps(r.to_dict()) + "\n" for r in results), encoding="utf-8"
            )

    papers = load_test_set(args.peerread_dir, args.venue)
    if args.limit:
        papers = papers[: args.limit]
    todo = [p for p in papers if p["paper_id"] not in already_done]
    logging.info("Loaded %d test paper(s), %d already done, %d to run", len(papers), len(already_done), len(todo))

    graph = build_review_graph()
    docling_parser = DoclingParser()

    with args.output.open("a", encoding="utf-8") as f:
        for i, paper in enumerate(todo, 1):
            logging.info("[%d/%d] Running %s: %s", i, len(todo), paper["paper_id"], paper["title"][:80])
            result = run_single_paper(graph, docling_parser, paper)
            results.append(result)
            f.write(json.dumps(result.to_dict()) + "\n")
            f.flush()
            status = f"ERROR: {result.error}" if result.error else (
                f"recommendation={result.final_recommendation} "
                f"predicted_accept={result.predicted_accept} ground_truth={result.ground_truth_accepted}"
            )
            logging.info("  -> %s (%.1fs)", status, result.elapsed_s or 0)

    metrics = compute_metrics(results)
    print(json.dumps(metrics, indent=2))
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
