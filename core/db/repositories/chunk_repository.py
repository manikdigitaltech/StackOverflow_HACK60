"""
Repository for the `chunks` table — the bridge between FAISS vector IDs
and the actual paper text/metadata in MySQL.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select

from core.db.models import Chunk, ChunkStatus


class ChunkRepository:
    def __init__(self, session: Session):
        self._session = session

    def insert_chunk(
        self,
        paper_id: int,
        faiss_id: int,
        chunk_text: str,
        embedding_model_version: str,
        section_type: Optional[str] = None,
    ) -> Chunk:
        chunk = Chunk(
            paper_id=paper_id,
            faiss_id=faiss_id,
            chunk_text=chunk_text,
            section_type=section_type,
            embedding_model_version=embedding_model_version,
            status=ChunkStatus.active,
        )
        self._session.add(chunk)
        self._session.flush()
        return chunk

    def get_active_by_faiss_ids(self, faiss_ids: List[int]) -> List[Chunk]:
        if not faiss_ids:
            return []
        return list(
            self._session.execute(
                select(Chunk).where(
                    Chunk.faiss_id.in_(faiss_ids),
                    Chunk.status == ChunkStatus.active,
                )
            ).scalars()
        )

    def mark_stale(self, chunk_ids: List[int]) -> None:
        if not chunk_ids:
            return
        chunks = self._session.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        ).scalars()
        for chunk in chunks:
            chunk.status = ChunkStatus.stale
        self._session.flush()
