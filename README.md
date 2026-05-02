# Text-To-SQL RAG System for Arabic Queries

A Retrieval-Augmented Generation (RAG) system that converts natural language questions — in both **Arabic and English** — into SQL queries. It uses the [AR-Spider dataset](https://github.com/YousefRadwan1/Text-To-Sql-Rag-System-for-Arabic-queries-/blob/main/AR_spider.jsonl), a bilingual (Arabic/English) variant of the Spider benchmark.

---

## Overview

This system accepts a natural language question, retrieves the most semantically similar examples from a FAISS vector index, and uses Google's Gemini LLM to generate the corresponding SQL query. The pipeline supports Arabic input natively via automatic language detection and an Arabic-capable embedding model.

**How it works:**

1. User types a question in Arabic or English.
2. The question is embedded using the `silma-ai/silma-embeddding-matryoshka-0.1` SentenceTransformer model.
3. FAISS retrieves the top-K most similar (question, SQL) pairs from the AR-Spider dataset.
4. The retrieved context and the original question are sent to Gemini (`gemini-1.5-flash`), which generates a clean SQL query.
5. The result is displayed in a Streamlit web UI or printed to the terminal.

---

## Project Structure

```
.
├── app.py            # Streamlit web application
├── rag_system.py     # Core RAG pipeline (orchestrates embedding, retrieval, LLM)
├── embedding.py      # SilmaEmbedding wrapper around SentenceTransformer
├── vector_db.py      # FAISS vector database (build, save, load, search)
├── llm.py            # GeminiLLM wrapper (language detection + SQL generation)
├── evaluation.py     # Batch evaluation script with exact-match scoring
└── AR_spider.jsonl   # Bilingual Arabic/English SQL dataset (AR-Spider)
```

---

## Requirements

- Python 3.8+
- A valid **Google Gemini API key**

Install dependencies:

```bash
pip install streamlit sentence-transformers faiss-cpu numpy google-generativeai jsonlines tqdm
```

> For GPU-accelerated FAISS, replace `faiss-cpu` with `faiss-gpu`.

---

## Setup & Usage

### 1. Set your Gemini API key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

The app will automatically build the FAISS index on first launch. Subsequent launches load the saved index from `indexes/faiss_index`.

### 3. Run from the command line

**Interactive mode:**
```bash
python rag_system.py --interactive
```

**Single question:**
```bash
python rag_system.py --question "ما هي أسماء الطلاب الذين تجاوزت درجاتهم 90؟"
```

**Available CLI flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--data` | Path to the JSONL dataset | `AR_spider.jsonl` |
| `--rebuild` | Force rebuild the FAISS index | `False` |
| `--top_k` | Number of retrieved examples | `5` |
| `--db_id` | Filter retrieval to a specific database | None |
| `--interactive` | Enable interactive Q&A loop | `False` |
| `--question` | Single question (non-interactive) | None |
| `--model` | SentenceTransformer model name | `silma-ai/silma-embeddding-matryoshka-0.1` |

### 4. Run evaluation

Evaluates the system on the first 100 examples from the dataset using exact SQL match:

```bash
python evaluation.py --data AR_spider.jsonl
```

---

## Dataset Format

`AR_spider.jsonl` is a JSONL file where each line contains:

```json
{
  "question": "What are the names of all singers?",
  "arabic": "ما هي أسماء جميع المغنين؟",
  "query": "SELECT name FROM singer",
  "db_id": "concert_singer"
}
```

---

## Module Details

### `embedding.py` — `SilmaEmbedding`
Wraps `sentence-transformers` to encode text using the `silma-ai/silma-embeddding-matryoshka-0.1` model, which supports Arabic and English. Each chunk is formatted as a combination of the English question, Arabic question, SQL query, and database ID before embedding.

### `vector_db.py` — `VectorDB`
Manages a FAISS `IndexFlatL2` index. Supports building from a JSONL file, saving/loading to disk, and searching by query with optional filtering by `db_id`.

### `llm.py` — `GeminiLLM`
Wraps the Gemini API (`gemini-1.5-flash`). Includes a `detect_language` method that checks the proportion of Arabic Unicode characters, and a `generate_response` method that builds a prompt with retrieved context and returns only the raw SQL string.

### `rag_system.py` — `RAGSystem`
Orchestrates the full pipeline: builds or loads the index, detects language, retrieves similar examples, and calls the LLM to generate SQL.

### `app.py`
Streamlit frontend with a text area for the question, a submit button, and a sidebar option to force-rebuild the index.

### `evaluation.py`
Batch evaluation over the first 100 dataset entries. Uses retry logic with exponential backoff to handle Gemini API quota errors (`ResourceExhausted`). Reports exact-match accuracy as a percentage.

---

## Notes

- The Gemini API key is read from the `GEMINI_API_KEY` environment variable. A fallback hardcoded key exists in the source but **should be replaced** before any public deployment.
- The FAISS index is saved under `indexes/faiss_index.idx` and `indexes/faiss_index_meta.json` after the first build.
- Language detection is heuristic-based: if more than 15% of input characters are in the Arabic Unicode range (`U+0600`–`U+06FF`), the input is treated as Arabic.
