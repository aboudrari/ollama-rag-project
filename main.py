
import ollama

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain self-attention in a Transformer using a simple example."
        }
    ]
)

answer = response["message"]["content"]

print(answer)