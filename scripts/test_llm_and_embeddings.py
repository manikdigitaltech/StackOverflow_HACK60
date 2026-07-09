"""
Step 4 test:
(a) sends a trivial prompt to Ollama through llm_provider + prompt_manager
(b) embeds a batch of texts with both Index A/B providers and confirms vector shape/dimension

Run with: python -m scripts.test_llm_and_embeddings
"""

from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.rag.embeddings.embedding_provider import BgeSmallEmbeddingProvider, Specter2EmbeddingProvider

print("--- Testing LLM (Ollama) ---")
llm = get_llm(json_mode=False)  # connectivity_test expects free text, not JSON
prompt_manager = PromptManager()
system, user = prompt_manager.render("connectivity_test", keyword="banana")

messages = [("system", system), ("human", user)] if system else [("human", user)]
response = llm.invoke(messages)
print(f"LLM response: {response.content}")

print("\n--- Testing Index A embeddings (bge-small-en-v1.5) ---")
bge = BgeSmallEmbeddingProvider()
doc_vectors = bge.embed(["This is a test sentence for embedding, as a document/passage."])
print(f"Document embedding dimension: {bge.dimension} (expect 384 for bge-small-en-v1.5)")
print(f"First 5 values: {doc_vectors[0][:5]}")

print("\n--- Testing Index B embeddings (specter2_base, falls back to bge-small if unavailable) ---")
specter = Specter2EmbeddingProvider()
lit_vectors = specter.embed(["A Test Paper Title[SEP]This is a test abstract about KV cache memory."])
print(f"Literature embedding dimension: {specter.dimension}")
print(f"First 5 values: {lit_vectors[0][:5]}")

print("\nStep 4 check complete. If all sections above printed with no errors, "
      "the LLM and embedding pipeline are ready for building the RAG indexes.")
