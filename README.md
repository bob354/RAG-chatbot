# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your PDF documents. Built with **LangChain**, **FAISS**, **Groq (LLaMA 3.1)**, and **Flask**.

---

## Architecture

```
User Question
     │
     ▼
┌──────────┐    ┌────────────────┐    ┌────────────┐
│  Flask   │───▶│  RAG Chain     │───▶│   Groq     │
│  Web UI  │    │  (LangChain)   │    │  LLaMA 3.1 │
└──────────┘    └───────┬────────┘    └────────────┘
                        │
                 ┌──────┴──────┐
                 │   FAISS     │
                 │ Vector Store│
                 └──────┬──────┘
                        │
                 ┌──────┴──────┐
                 │    PDFs     │
                 │ (documents/)│
                 └─────────────┘
```

**How it works:**

1. PDFs in the `documents/` folder are loaded and split into chunks.
2. Chunks are embedded using a HuggingFace sentence-transformer model and stored in a FAISS vector store.
3. When a user asks a question, the most relevant chunks are retrieved via cosine similarity.
4. The retrieved context + question are sent to the Groq-hosted LLaMA 3.1 model to generate an answer.
5. The chatbot only answers based on the provided documents — no hallucination.

---

## Prerequisites

- **Python 3.10+**
- **A Groq API key** — get one for free at [console.groq.com](https://console.groq.com/)
- **PDF documents** you want the chatbot to answer questions about

---

## Setup Guide

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd prj-chatbot-rag
```

## Setup

### Option 1: venv (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Option 2: conda (if you have Anaconda)
```bash
conda create -n rag_chatbot python=3.10 -y
conda activate rag_chatbot
pip install -r requirements.txt
```

This installs:

| Package                   | Purpose                                |
| ------------------------- | -------------------------------------- |
| `Flask`                   | Web server & API                       |
| `langchain_community`     | Document loaders, FAISS vector store   |
| `langchain_core`          | Prompts, output parsers, runnables     |
| `langchain_groq`          | Groq LLM integration                  |
| `langchain_huggingface`   | HuggingFace embeddings                 |
| `langchain_text_splitters` | Text chunking                         |
| `python-dotenv`           | Environment variable management        |

> **Note:** The HuggingFace embedding model (~500 MB) will be downloaded automatically on first run.

### 4. Configure environment variables

Create a `.env` file in the project root (or edit the existing one):

```env
# ── Embedding Model ──
HUGGINGFACE_EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── LLM (Groq) ──
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL="llama-3.1-8b-instant"
```

| Variable                       | Description                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `HUGGINGFACE_EMBEDDING_MODEL`  | The sentence-transformer model used to create embeddings. The default works well for multilingual text. |
| `GROQ_API_KEY`                 | Your Groq API key. Get one at [console.groq.com](https://console.groq.com/).                         |
| `GROQ_MODEL`                   | The LLM model hosted on Groq. `llama-3.1-8b-instant` is fast and free-tier friendly.                 |

### 5. Add your PDF documents

Create a `documents` folder in the project root and place your PDF files inside it:

```text
documents/
├── your-file-1.pdf
├── your-file-2.pdf
└── ...
```

The chatbot will automatically load **all `.pdf` files** from this directory on startup.

### 6. Run the application

```bash
python app.py
```

The server will start at: **http://localhost:5000**

Open this URL in your browser to start chatting!

---

## Usage

### Web Interface

1. Open **http://localhost:5000** in your browser.
2. Type a question about your documents in the input field.
3. Press **Enter** (or click the send button) to submit.
4. The chatbot will retrieve relevant passages and generate an answer with source citations.

### Command Line

You can also use the chatbot directly from the terminal:

```bash
python chatbot.py
```

This will prompt you for a question and print the answer to the console.

### API Endpoint

Send a POST request to `/api/chat`:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

**Response:**

```json
{
  "answer": "Based on the provided documents, ..."
}
```

### Source Search Endpoint

Use `/api/search` to get ranked snippets from the indexed documents, similar to NotebookLM-style source search:

```bash
curl -X POST http://localhost:5000/api/search \
    -H "Content-Type: application/json" \
    -d '{"query": "main topic", "active_docs": ["paper.pdf"], "max_results": 8}'
```

**Response:**

```json
{
    "query": "main topic",
    "count": 2,
    "results": [
        {
            "source": "paper.pdf",
            "page": 3,
            "snippet": "...",
            "score": 0.91
        }
    ]
}
```

---

## Project Structure

```
prj-chatbot-rag/
├── app.py              # Flask web server (routes & API)
├── chatbot.py          # RAG pipeline (loader → splitter → embeddings → retriever → LLM)
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API keys, model names)
├── documents/          # Place your PDF files here
│   └── *.pdf
└── templates/
    └── index.html      # Chat web interface
```

---

## Configuration & Tuning

### Chunk Size & Overlap

In `chatbot.py`, you can adjust how documents are split:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,      # Max characters per chunk (increase for more context)
    chunk_overlap=200,    # Overlap between chunks (helps preserve context at boundaries)
)
```

### Number of Retrieved Documents

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,  # Number of document chunks to retrieve per query
    }
)
```

### LLM Temperature

```python
llm = ChatGroq(
    temperature=0  # 0 = deterministic, higher = more creative
)
```

### Alternative Embedding Models

Update `HUGGINGFACE_EMBEDDING_MODEL` in `.env`:

| Model                                                           | Size    | Best For           |
| --------------------------------------------------------------- | ------- | -------------------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`   | ~470 MB | Multilingual (default) |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`   | ~1.1 GB | Higher accuracy    |
| `sentence-transformers/all-MiniLM-L6-v2`                        | ~90 MB  | English only, fast |

---

## Troubleshooting

| Issue                                  | Solution                                                                                     |
| -------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError`                  | Make sure your virtual environment is activated and run `pip install -r requirements.txt`     |
| `GROQ_API_KEY` errors                  | Verify your `.env` file has a valid Groq API key                                             |
| Slow first startup                     | The embedding model is being downloaded (~500 MB). Subsequent runs will use the cached model. |
| `No documents found` / empty responses | Ensure `.pdf` files are placed in the `documents/` folder                                    |
| Port 5000 already in use               | Change the port in `app.py`: `app.run(port=5001)`                                            |
| Out of memory                          | Use a smaller embedding model (e.g., `all-MiniLM-L6-v2`) or reduce `chunk_size`              |

---

## License

This project is for educational purposes.
