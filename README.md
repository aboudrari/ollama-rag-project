# Local RAG System with Ollama

A local Retrieval-Augmented Generation (RAG) system built with Python and Ollama. It retrieves relevant information from PDF and text documents and uses a local language model to answer questions based on the retrieved context.

## Features

- Load PDF and TXT documents.
- Split documents into smaller text chunks.
- Generate embeddings using `nomic-embed-text`.
- Retrieve relevant chunks using cosine similarity.
- Generate answers with `qwen2.5:3b`.
- Return source document names with answers.
- Respond with `I don't know.` when the retrieved context does not provide enough information.

## How It Works

1. **Load Documents:** Read PDF and TXT files from the documents directory.
2. **Chunk Text:** Split document text into overlapping chunks.
3. **Generate Embeddings:** Convert chunks into numerical vectors using `nomic-embed-text`.
4. **Retrieve Context:** Compare the question embedding with document embeddings using cosine similarity.
5. **Generate Answer:** Send the question and retrieved context to `qwen2.5:3b`.
6. **Return Results:** Display the answer and source document names.

## Requirements

- Python 3.11
- [Ollama](https://ollama.com/)
- The `qwen2.5:3b` language model
- The `nomic-embed-text` embedding model

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aboudrari/ollama-rag-project.git
cd ollama-rag-project
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

Install the packages required by the project:

```powershell
pip install ollama pypdf
```

> If the project has additional imports, install those dependencies as well.

### 4. Download the Ollama models

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

## Add Your Documents

Place your PDF and TXT files in the project's documents directory.

The application processes supported documents and creates text chunks for retrieval.

## Run the Application

```powershell
python main.py
```

Enter a question when prompted. Type `exit` to quit.

## Example

```text
Your question: What is the minimum CGPA required for graduation?

Answer: 2.00
Sources: ['ciu_exam_regulations.pdf']
```

Answers depend on the content and quality of the supplied documents and the retrieval results.

## Project Structure

```text
ollama-rag-project/
├── data/
│   └── documents/
├── src/
│   ├── retriever.py
│   └── rag.py
├── main.py
├── evaluate_rag.py
└── README.md
```

## Limitations

- Answer quality depends on document content, chunking, and retrieval quality.
- Local inference speed depends on available hardware.
- The system may return `I don't know.` when it cannot find sufficient evidence in the retrieved context.
- Source names identify documents from which chunks were retrieved; they do not necessarily prove that every answer is supported by those documents.

## Future Improvements

- Improve retrieval and answer evaluation.
- Add a simple user interface.
- Improve document citation and source display.

## License

Choose a license before distributing the project.
