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
You are a question-answering assistant.

Answer the question using only the information explicitly
provided in the context.

Important rules:
- The context may discuss the topic without containing the answer.
- Do not guess, infer missing facts, or use outside knowledge.
- If the context does not explicitly answer the question,
  respond exactly with: I don't know.
- If the context provides the answer, answer briefly.

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