"""Structure-aware chunker for Index A (Paper-RAG).

Two-pass design: first split the parsed paper on its own section boundaries
(abstract, intro, method, ...), then, within each section, split long spans
into ~300-500 token windows with overlap. Splitting on structure BEFORE size
is what keeps a chunk's `section` metadata meaningful - a naive fixed-window
split over the raw text would blend method text into results text at
boundaries and make `section_filter` retrieval useless.
"""
from __future__ import annotations

import re
from typing import Any

from core.config.rag_settings import RAG_SETTINGS
from core.rag.models import Chunk

# canonical section names accepted by Chunk.section; anything unmatched -> "other"
_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("abstract", r"abstract"),
    ("introduction", r"introduction|^intro\b"),
    ("related_work", r"related.work|background|prior.work"),
    ("method", r"method|approach|model|architecture|proposed"),
    ("experiments", r"experiment|setup|evaluation|implementation"),
    ("results", r"result|finding|analysis"),
    ("conclusion", r"conclusion|discussion|future.work|summary"),
]

_tokenizer = None
_tokenizer_failed = False


def _get_tokenizer():
    """Lazy tokenizer for RAG_SETTINGS.paper_index.embedding_model; None if
    transformers is unavailable (falls back to whitespace counting so the
    chunker never hard-fails - fail-soft, per the subsystem contract)."""
    global _tokenizer, _tokenizer_failed
    if _tokenizer is None and not _tokenizer_failed:
        try:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(RAG_SETTINGS.paper_index.embedding_model)
        except Exception:
            _tokenizer_failed = True
    return _tokenizer


def normalize_section_name(heading: str) -> str:
    """Map a raw section heading to one of the canonical `Chunk.section` values."""
    h = (heading or "").lower().strip()
    for canonical, pattern in _SECTION_PATTERNS:
        if re.search(pattern, h):
            return canonical
    return "other"


def _looks_like_table(paragraph: str) -> bool:
    """A paragraph is a table if multiple lines carry multiple column separators."""
    table_lines = [ln for ln in paragraph.splitlines() if ln.count("|") >= 2]
    return len(table_lines) >= 2


def chunk_paper(paper_id: str, parsed_paper: Any) -> list[Chunk]:
    """Split a parsed paper into section-aware, size-bounded chunks.

    Args:
        paper_id: identifier of the paper under review, propagated onto
            every produced `Chunk` so downstream retrieval can scope by paper.
        parsed_paper: the structured paper object produced by the parsing
            layer. Accepted shapes: a dict with a "sections" mapping of
            {section_name: text} (see tests/fixtures/sample_paper.json), an
            object with a `.sections` mapping attribute, or an iterable of
            (section_name, text) pairs.

    Returns:
        Ordered list of `Chunk`, each tagged with `section`, `para_idx`,
        and `has_table`.
    """
    if isinstance(parsed_paper, dict):
        sections = parsed_paper.get("sections", {})
    elif hasattr(parsed_paper, "sections"):
        sections = parsed_paper.sections
    else:
        sections = parsed_paper
    items = sections.items() if hasattr(sections, "items") else list(sections)

    max_tokens = RAG_SETTINGS.chunking.max_tokens
    chunks: list[Chunk] = []
    for raw_name, text in items:
        section = normalize_section_name(str(raw_name))
        text = (text or "").strip()
        if not text:
            continue
        para_idx = 0
        # Pass 1 within the section: peel off table paragraphs so they stay
        # intact (splitting a table's cells across chunks destroys meaning),
        # accumulate everything else into one prose span.
        prose_parts: list[str] = []
        pending: list[tuple[str, bool]] = []  # (text, has_table) in document order
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            if _looks_like_table(para):
                if prose_parts:
                    pending.append(("\n\n".join(prose_parts), False))
                    prose_parts = []
                pending.append((para, True))
            else:
                prose_parts.append(para)
        if prose_parts:
            pending.append(("\n\n".join(prose_parts), False))

        # Pass 2: size-bound the prose spans; tables bypass the size limit.
        for span, has_table in pending:
            windows = [span] if has_table or _count_tokens(span) <= max_tokens \
                else _split_section_into_windows(span, section)
            for w in windows:
                chunks.append(Chunk(
                    chunk_id=f"{paper_id}:{section}:{para_idx}",
                    paper_id=paper_id, section=section, para_idx=para_idx,
                    text=w, has_table=has_table, token_count=_count_tokens(w),
                ))
                para_idx += 1
    return chunks


def _split_section_into_windows(section_text: str, section_name: str) -> list[str]:
    """Token-bounded sliding-window split of a single section's text.

    Args:
        section_text: raw text of one section.
        section_name: name of the section being split, for error messages.

    Returns:
        List of overlapping text windows, each within
        [min_tokens, max_tokens] tokens (see `ChunkingSettings`).
    """
    cfg = RAG_SETTINGS.chunking
    tok = _get_tokenizer()
    if tok is not None:
        ids = tok.encode(section_text, add_special_tokens=False)
        decode = lambda window: tok.decode(window).strip()
    else:  # whitespace fallback - same accounting as _count_tokens
        ids = section_text.split()
        decode = lambda window: " ".join(window)

    if not ids:
        raise ValueError(f"section {section_name!r} produced no tokens")
    step = cfg.target_tokens - cfg.overlap_tokens
    windows: list[list] = []
    for start in range(0, len(ids), step):
        window = ids[start: start + cfg.target_tokens]
        if windows and len(window) < cfg.min_tokens:
            # tail too small to stand alone: extend backwards so every window
            # stays within [min_tokens, max_tokens] instead of emitting a runt
            window = ids[max(0, len(ids) - cfg.target_tokens):]
            windows[-1] = window
            break
        windows.append(window)
        if start + cfg.target_tokens >= len(ids):
            break
    return [decode(w) for w in windows]


def _count_tokens(text: str) -> int:
    """Return the token count of `text` under the embedding model's tokenizer.

    Kept as its own function so the chunker and the embedding provider agree
    on what "300-500 tokens" means without duplicating tokenizer setup.
    Falls back to whitespace counting if the tokenizer cannot load.
    """
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text.split())
