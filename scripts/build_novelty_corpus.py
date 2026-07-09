"""
Converts PeerRead `reviews/*.json` (title/abstract/accepted only) into the
per-paper JSON files `data/novelty_corpus/` expects (title/abstract/sections/
references/year) -- populates the embedding-based Novelty Evaluation Agent's
background corpus with real ICLR-2017 papers instead of the 2 toy seed papers.

Follows the same train+dev-only, test-excluded split policy as
core/rag/ingestion/build_corpus.py, so the eventual PeerRead test-split
evaluation harness never has its ground-truth papers leaking into an agent's
own background corpus.

Run with:
    python -m scripts.build_novelty_corpus --peerread-dir data/peerread_raw --venue iclr_2017
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CORPUS_SPLITS = ("train", "dev")
HELD_OUT_SPLITS = ("test",)


def _load_ids_and_records(peerread_dir: Path, venues: list[str], splits: tuple[str, ...]) -> list[dict]:
    records = []
    for venue in venues:
        for split in splits:
            reviews_dir = peerread_dir / venue / split / "reviews"
            if not reviews_dir.is_dir():
                continue
            for json_path in sorted(reviews_dir.glob("*.json")):
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if not data.get("abstract"):
                    continue
                data["_venue"] = venue
                records.append(data)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peerread-dir", type=Path, required=True, help="Path to PeerRead's data/ directory")
    parser.add_argument("--venue", action="append", dest="venues", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/novelty_corpus"))
    args = parser.parse_args()

    held_out_ids = {r["id"] for r in _load_ids_and_records(args.peerread_dir, args.venues, HELD_OUT_SPLITS)}
    records = _load_ids_and_records(args.peerread_dir, args.venues, CORPUS_SPLITS)
    records = [r for r in records if r["id"] not in held_out_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for r in records:
        out_path = args.output_dir / f"{r['_venue']}_{r['id']}.json"
        out_path.write_text(
            json.dumps(
                {
                    "title": r.get("title", ""),
                    "abstract": r.get("abstract", ""),
                    "sections": [],
                    "references": [],
                    "year": int(r["_venue"].split("_")[-1]) if r["_venue"].split("_")[-1].isdigit() else None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        written += 1

    print(f"Wrote {written} novelty-corpus papers to {args.output_dir} "
          f"(excluded {len(held_out_ids)} held-out test-split ids)")


if __name__ == "__main__":
    main()
