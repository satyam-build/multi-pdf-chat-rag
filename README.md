# multi-pdf-chat-rag
RAG-powered chatbot to query multiple PDFs using natural language. Built with LangChain, FAISS, HuggingFace embeddings &amp; Groq's Llama 3. Features vectorized memory — only semantically relevant chat turns are sent to the LLM, not the full history.

# Multi-PDFs ChatApp AI Agent 📚

Query multiple PDF documents through a conversational interface powered by RAG.
Upload any PDFs, ask questions across all of them, and get grounded answers —
no hallucinations, no guessing.

## Tech Stack
- **LLM** — Llama 3.3 70B via Groq API (sub-second inference)
- **Embeddings** — HuggingFace all-MiniLM-L6-v2 (runs locally, no API limit)
- **Vector Store** — FAISS (PDF chunks) + FAISS (vectorized chat memory)
- **Framework** — LangChain Classic
- **UI** — Streamlit

## Key Features
- 📄 Upload and query multiple PDFs simultaneously
- 🧠 Vectorized memory — embeds each Q&A turn and retrieves only the top-k
  relevant past exchanges per query, keeping token usage flat regardless of
  conversation length
- 🚫 Hallucination guard — LLM is instructed to only answer from retrieved context
