"""
Verifies Step 5's result: FAISS index exists and has vectors, MySQL has
matching papers/chunks rows, and a sample similarity search returns
sensible results.

Run with: python -m scripts.verify_ingestion
"""

from sqlalchemy import select, func
from core.db.session import get_session
from core.db.models import Paper, Chunk
from core.rag.vectorstore.faiss_index_manager import load_or_create_index
from core.rag.embeddings.embedding_provider import EmbeddingProvider

print("--- Verifying FAISS index ---")
store = load_or_create_index()
print(f"FAISS index total vectors: {store.ntotal}")

print("\n--- Verifying MySQL rows ---")
with get_session() as session:
    paper_count = session.execute(select(func.count()).select_from(Paper)).scalar()
    chunk_count = session.execute(select(func.count()).select_from(Chunk)).scalar()
    print(f"papers table: {paper_count} rows")
    print(f"chunks table: {chunk_count} rows")

    print("\n--- Sample papers ---")
    sample_papers = session.execute(select(Paper).limit(5)).scalars().all()
    for p in sample_papers:
        print(f"  [{p.id}] {p.title[:80]}")

    print("\n--- Checking for duplicate titles (e.g. arXiv version-suffix bug) ---")
    dup_check = session.execute(
        select(Paper.title, func.count().label("cnt"))
        .group_by(Paper.title)
        .having(func.count() > 1)
    ).all()
    if dup_check:
        print(f"WARNING: {len(dup_check)} duplicate title(s) found:")
        for title, cnt in dup_check:
            print(f"  x{cnt}: {title[:80]}")
    else:
        print("No duplicate titles found -- good.")

print("\n--- Test similarity search (FAISS only, no MySQL join yet -- that's Step 6) ---")
embedder = EmbeddingProvider()
query_vector = embedder.embed_query("memory efficient attention mechanisms for transformers")
results = store.similarity_search(query_vector, k=3)
print(f"Top {len(results)} matches (faiss_id, similarity score):")
for faiss_id, score in results:
    print(f"  faiss_id={faiss_id}, score={score:.4f}")

print("\nIf papers/chunks counts are both > 0, FAISS has vectors, and the "
      "similarity search returned plausible-looking faiss_ids with scores "
      "between roughly 0 and 1 -- Step 5 is working.")
