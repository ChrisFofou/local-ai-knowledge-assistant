from dataclasses import dataclass, field
from typing import Any

from knowledge_assistant.embeddings.encoder import EmbeddingEncoder
from knowledge_assistant.vector_store.store import ChromaVectorStore


@dataclass
class RetrievedChunk:
    """Represent a document chunk returned by semantic retrieval."""

    chunk_id: str
    text: str
    source: str
    document_id: str
    chunk_index: int
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalService:
    """Coordinate query embedding and semantic vector search."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        vector_store: ChromaVectorStore,
    ):
        self.encoder = encoder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant document chunks for a query."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.encoder.encode_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        retrieved_chunks = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            metadata = metadata or {}

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=metadata.get("source", "unknown"),
                    document_id=metadata.get("document_id", "unknown"),
                    chunk_index=metadata.get("chunk_index", -1),
                    distance=float(distance),
                    metadata=metadata,
                )
            )

        return retrieved_chunks
