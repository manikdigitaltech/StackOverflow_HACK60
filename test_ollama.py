import ollama

prompt = """
You are a Novelty Review Agent for scientific papers.

Important:
RAG means Retrieval-Augmented Generation.
Do not use any other meaning of RAG.

Explain how RAG helps compare a submitted research paper
with previous literature.
Explain in simple English.
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])