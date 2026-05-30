# Retrieval-Augmented Generation (RAG) — Project README

This repository contains simple pipelines for building a RAG system.

## 1. Ingestion

Purpose: prepare and encode source documents into a vector store so they can be retrieved during generation.

Typical steps:

- Collect sources: gather documents (text, PDFs, web pages) into a local folder (see `docs/`).
- Text extraction: convert files to plain text and clean (remove boilerplate, normalize whitespace).
- Chunking: split long documents into smaller passages (e.g., 200–1000 tokens) with overlap to preserve context.
- Embedding: compute dense vector embeddings for each chunk using a chosen embedding model.
- Metadata: attach metadata (source, doc id, chunk index, title, URL, timestamps) to each vector.
- Persist: store vectors + metadata in a vector database (e.g., Chroma) or any other vector store.

Reference file: `1_ingestion_pipeline.py` implements a basic ingestion flow.

In short:

- Document loading (PDF, DOCX, TXT -> Text)
- Text Chunking (long text -> smaller pieces)
- Embedding (text chunks -> vectors)
- Storage (vectos -> vector db)

## 2. Retrieval

Purpose: given a user query or prompt, find the most relevant document chunks from the vector store to condition generation.

Typical steps:

- Query embedding: encode the user query into a vector with the same embedding model used at ingestion.
- Similarity search: query the vector store to retrieve top-k nearest neighbors (by cosine or dot product).
- Filtering & reranking: optionally filter results by metadata (date, source) and rerank with a cross-encoder or BM25 hybrid approach.
- Context assembly: assemble retrieved chunks into a context window for the generator, respecting token limits and ordering.
- Prompt construction: craft the final prompt that includes retrieved context plus user question or instruction.

Reference file: `2_retrieval_pipeline.py` contains retrieval and prompt-assembly logic.

## Notes & Tips

- Keep embedding model consistent between ingestion and retrieval.
- Tune chunk size and overlap to balance relevance and token usage.
- Persist metadata to support provenance and safe filtering.
- Consider hybrid retrieval (dense + sparse) and reranking for higher accuracy.

Chunking us the 2nd critical step - it determines how your content gets divided for retrieval.

# Top 5 chunking strategies

1. CharacterTextSplitter (Beyond basic chunk_size)
   - Custom separatos (Split on specific patterns)
   - Still useful for simple, uniform documents or when speed matters most

2. RecursiveCharacterTextSplitter (Upgrade from CharacterTextSplitter)
   - Tries to split at natural boundaries (paragraphs, sentences, words)
   - Falls back gracefully if chunk is too big
   - Preserves more context than basic splitting

3. Document-Specific Splitting (Respects document structure)
   - PDF: Splits by pages, sections, headers
   - Markdown: Splits by headers, code blocks, lists
   - Each document type gets appropriate treatment

4. Semantic Splitting (Content-aware boundaries)
   - Uses embeddings to detect topic shifts
   - Keeps related concepts together
   - Splits when meaning changes, not just by size
   - More intelligent but computationally expensive

5. Agentic Splitting (AI-powered chunking)
   - LLM analyzes content and decides optimal splits
   - Can understand complex relationships
   - Adapts to content type autmatically
   - Most sophisticated but slowest/most expensive
