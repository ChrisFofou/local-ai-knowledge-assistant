# System Architecture

## 1. Overview

Local AI Knowledge Assistant is a locally hosted Retrieval-Augmented Generation
(RAG) application. It allows users to query a multilingual document collection
and receive answers grounded in retrieved document content.

## 2. Architectural Components

### API Layer
FastAPI exposes endpoints for document ingestion and question answering.

### Ingestion Pipeline
Responsible for:
- Loading supported document formats.
- Extracting and normalizing text.
- Splitting documents into manageable chunks.
- Attaching document metadata.

### Embedding Provider
Uses Sentence Transformers to convert documents and queries into dense vectors.
The initial model is:

paraphrase-multilingual-MiniLM-L12-v2

### Vector Store
ChromaDB provides persistent vector storage and similarity search.

### Retrieval
Finds the most relevant document chunks for a user query.

### Generation
Ollama hosts the local language model that generates answers using retrieved
context.

## 3. Data Flow

### Document Ingestion

Document -> Loader -> Text Extraction -> Chunking -> Embeddings -> ChromaDB

### Question Answering

Question -> Query Embedding -> Vector Search -> Context Assembly ->
Ollama -> Grounded Answer

## 4. Design Principles

- Modular components with clear responsibilities.
- Local-first execution and data privacy.
- Replaceable model and storage providers.
- Explicit configuration instead of hardcoded settings.
- Testable components and reproducible experiments.
- Answers grounded in retrieved source material.

## 5. Initial Technology Choices

| Component | Technology |
|---|---|
| API | FastAPI |
| ML Framework | PyTorch |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Language Model Runtime | Ollama |
| Document Extraction | pypdf |
| Testing | pytest |
| Deployment | Docker |

## 6. Future Improvements

- Support additional document formats.
- Add hybrid retrieval and reranking.
- Implement retrieval and answer quality evaluation.
- Add document-level citations.
- Add authentication and access control.
- Add observability and structured logging.
