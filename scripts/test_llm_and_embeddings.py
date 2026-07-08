"""
Step 4 test:
(a) sends a trivial prompt to Ollama through llm_provider + prompt_manager
(b) embeds a sentence and confirms vector shape/dimension
(c) confirms embed_query() vs embed() both work (retrieval quality detail)

Run with: python -m scripts.test_llm_and_embeddings
"""

from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.rag.embeddings.embedding_provider import EmbeddingProvider

print("--- Testing LLM (Ollama) ---")
llm = get_llm()
prompt_manager = PromptManager()
system, user = prompt_manager.render("connectivity_test", keyword="banana")

messages = [("system", system), ("human", user)] if system else [("human", user)]
response = llm.invoke(messages)
print(f"LLM response: {response.content}")

print("\n--- Testing embeddings ---")
embedder = EmbeddingProvider()

doc_vector = embedder.embed("This is a test sentence for embedding, as a document/passage.")
print(f"Document embedding dimension: {len(doc_vector)} (expect 1024 for bge-large-en-v1.5)")
print(f"First 5 values: {doc_vector[:5]}")

query_vector = embedder.embed_query("What methods reduce KV cache memory?")
print(f"Query embedding dimension: {len(query_vector)} (should match document dimension)")

print("\nStep 4 check complete. If both sections above printed with no errors, "
      "the LLM and embedding pipeline are ready for Step 5 (RAG ingestion).")
