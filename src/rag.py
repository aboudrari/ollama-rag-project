import ollama

from retriever import retrieve_relevant_chunks


def rag(question, chunks, embeddings, top_k=1, threshold=0.6):
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
Answer the question using only the provided context.
If the answer is not contained in the context, say:
"I don't know."

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
    )

    answer = response["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": sources,
    }