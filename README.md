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

If you want, I can add example commands to run the pipelines or expand the README with architecture diagrams.
