from collections.abc import Sequence
from typing import Any

import chromadb
import numpy as np

from knowledge_assistant.ingestion.chunker import DocumentChunk


class ChromaVectorStore:
    """Store and retrieve document embeddings using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "knowledge_assistant",
        host: str = "localhost",
        port: int = 8000,
        client: Any | None = None,
    ):
        self.client = client or chromadb.HttpClient(host=host, port=port)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Insert or update document chunks and their embeddings."""
        if not chunks:
            raise ValueError("Cannot store an empty chunk collection")

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")

        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values")

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=vectors.tolist(),
            metadatas=[
                {
                    key: value
                    for key, value in chunk.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                }
                | {
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find the most similar document chunks."""
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim != 1 or query.size == 0:
            raise ValueError("Query embedding must be a non-empty one-dimensional vector")

        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite values")

        if self.collection.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(n_results, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        self.collection.delete(where={"document_id": document_id})
