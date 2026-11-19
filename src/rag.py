
import ollama

from retriever import retrieve_relevant_chunks


def rag(question, chunks, embeddings, top_k=2, threshold=0.6):
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

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a precise document question-answering assistant.

Your task is to answer the user's question using ONLY
the information provided in the context.

Rules:
- Read all the provided context before answering.
- Identify the exact information requested.
- Pay close attention to numbers, conditions, and exceptions.
- For classification questions, look for explicit statements
  that define which categories qualify or do not qualify.
- Treat explicit classification statements as authoritative.
- Do not determine whether a category qualifies from its numerical
  value, coefficient, or position in a table.
- When a table is followed by an explanatory statement, read
  that statement carefully before answering.
- Do not use external knowledge.
- Do not guess or invent information.
- If the answer is not explicitly supported by the context,
  respond exactly: I don't know.
- Keep the answer brief and direct.

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
        options={
            "temperature": 0,
        },
    )

    answer = response["message"]["content"].strip()

    normalized_answer = " ".join(answer.lower().split())
    normalized_answer = normalized_answer.strip(" .!?\"'")

    if normalized_answer.startswith("i don't know"):
        return {
            "answer": "I don't know.",
            "sources": [],
        }

    sources = sorted({
        chunk["source"]
        for chunk in retrieved_chunks
    })

    return {
        "answer": answer,
        "sources": sources,
    }
