# Local AI Knowledge Assistant

A multilingual, locally hosted AI knowledge assistant designed to retrieve information from personal documents and generate answers grounded in their content.

The project explores Retrieval-Augmented Generation (RAG), semantic search, local language models, and production-oriented AI application architecture.

## Objectives

- Ingest documents in multiple languages.
- Convert documents into vector embeddings.
- Store and search embeddings using ChromaDB.
- Retrieve relevant information for user questions.
- Generate context-aware answers using locally hosted LLMs.
- Provide a practical API for interacting with the system.
- Evaluate retrieval quality and answer reliability.

## Architecture

Planned architecture:

```text
User
 |
 v
FastAPI Application
 |
 v
RAG Pipeline
 |--------------------|
 v                    v
Embedding Model       Ollama LLM
 |                    ^
 v                    |
ChromaDB -------------|
 |
 v
Retrieved Context
 |
 v
Grounded Answer

## Technology Stack

- Python
- PyTorch
- Sentence Transformers
- Hugging Face
- Ollama
- ChromaDB
- FastAPI
- Docker
- JupyterLab
## Current Development Status
Completed
- Ubuntu AI development environment configured.
- NVIDIA GPU acceleration verified.
- PyTorch CUDA execution tested successfully.
- Ollama installed and local language models tested.
- ChromaDB deployed with Docker.
- Multilingual embedding model tested.
- Vector search successfully tested with English and German documents.
Planned
- Document ingestion pipeline.
- Persistent knowledge base.
- Retrieval and generation pipeline.
- FastAPI endpoints.
- Evaluation framework.
- Dockerized application deployment.
Project Structure
local-ai-knowledge-assistant/
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
├── notebooks/
├── src/
├── tests/
├── models/
├── README.md
└── .gitignore

Development Environment
Python virtual environment:
source ~/venvs/ai-engineering/bin/activate

Run JupyterLab:
jupyter lab

Development Philosophy
The objective is to build a practical, maintainable AI application while documenting architectural decisions, experiments, evaluation results, and lessons learned.
License
MIT License
