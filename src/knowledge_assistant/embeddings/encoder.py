from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingEncoder:
    """Generate normalized multilingual text embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode multiple document chunks into normalized vectors."""
        if not texts:
            raise ValueError("Cannot encode an empty document collection")

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts cannot be empty")

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a normalized vector."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(embedding, dtype=np.float32)
