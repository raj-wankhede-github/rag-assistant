# RAG Assistant

A local command-line RAG assistant over your own PDFs, text and Markdown files.

Documents are loaded from a folder on your machine, chunked, embedded locally with
a sentence-transformer model and stored in a Chroma database next to the code. Only
the question and the retrieved passages are sent to a language model — the files
themselves never leave your machine.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- An API key for the chat model you use (OpenAI by default, Anthropic supported)

## Setup

```bash
uv sync
cp .env.example .env        # then fill in your key(s)
mkdir -p src/docs           # put your PDF / .txt / .md files here
uv run rag-assistant
```

`.env` holds:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Only the key for the model you actually use is required. `how_to_run.txt` has the
same steps including the Windows PowerShell one-liner for installing uv.

## Using it

```
Loading documents...
Loaded 12 documents
Split into 143 chunks
Vector store created

RAG Assistant ready. Type your questions (type 'quit' to exit):

You: what is the notice period in the contract?
```

Type `quit`, `exit` or `q` to leave. If `src/docs` is empty the assistant says so
and exits instead of answering from nothing.

## How it works

| Stage | Choice |
| --- | --- |
| Loading | `PyPDFLoader` for PDFs, `TextLoader` for `.txt` and `.md`, read from `src/docs` |
| Chunking | `RecursiveCharacterTextSplitter`, 1000 characters with 200 overlap |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace — runs locally, no API call |
| Vector store | Chroma, persisted to `./chroma_db` |
| Retrieval | Top 4 chunks per question |
| Generation | `gpt-4o-mini` at temperature 0.3 (the `ChatAnthropic` line in `build_chain` is the drop-in alternative) |
| Memory | Last 10 turns; follow-up questions are rewritten into standalone questions before retrieval |

The follow-up rewriting matters: it means "and what about the second one?" gets
turned into a self-contained question before it hits the retriever, instead of
retrieving on a fragment.

## Layout

```
src/rag_assistant/rag.py   loading, chunking, vector store, chain, CLI loop
src/docs/                  your documents (not committed)
chroma_db/                 persisted vectors (created on first run)
.env.example               required API keys
how_to_run.txt             step-by-step setup notes
```

## Notes and known issues

- The vector store is rebuilt from `src/docs` on every start; `chroma_db/` grows
  across runs. Delete it to start from a clean index.
- Switching the chat model between OpenAI and Anthropic is a one-line change in
  `build_chain` — retrieval and embeddings are unaffected, so no re-indexing is
  needed.
- The answer-printing path in `main` reads `response_text['answer']`, but the chain
  ends in `StrOutputParser` and returns a plain string. Printing the response
  directly (and appending that string to the history) is the fix.
