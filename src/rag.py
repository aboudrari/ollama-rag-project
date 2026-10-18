
import ollama

from retriever import retrieve_relevant_chunks


def rag(question, chunks, embeddings, top_k=3, threshold=0.6):
    """Answer a question using retrieved document context."""

    retrieved_chunks = retrieve_relevant_chunks(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        top_k=top_k,
        threshold=threshold,
    )

    if not retrieved_chunks:
        return {
            "answer": "I don't know.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for chunk in retrieved_chunks:
        context_parts.append(chunk["text"])
        sources.append(chunk["source"])

    context = "\n\n".join(context_parts)
    sources = sorted(set(sources))

    prompt = f"""
You are a precise question-answering assistant.

Answer the question using only information explicitly provided
in the context.

Rules:
- Read the entire context carefully before answering.
- Identify the exact fact requested by the question.
- Pay attention to distinctions such as passing versus failing,
  minimum versus maximum, and included versus excluded values.
- Do not confuse a value mentioned near the answer with the answer itself.
- Do not guess, infer missing facts, or use outside knowledge.
- If the context does not explicitly contain enough information
  to answer the question, respond exactly with: I don't know.
- Keep the answer brief and directly answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={"temperature": 0},
    )

    answer = response["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": sources,
    }
