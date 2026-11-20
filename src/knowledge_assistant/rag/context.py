"""Context assembly for retrieval-augmented generation."""

from dataclasses import dataclass, field
from math import ceil
from typing import Protocol

from knowledge_assistant.retrieval.service import RetrievedChunk


class TokenCounter(Protocol):
    """Interface for estimating the number of tokens in text."""

    def count(self, text: str) -> int:
        """Return the estimated token count for text."""


class CharacterTokenCounter:
    """Estimate tokens using approximately three characters per token."""

    def count(self, text: str) -> int:
        return ceil(len(text) / 3)


@dataclass
class ContextEntry:
    """A retrieved chunk prepared for inclusion in the LLM context."""

    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    chunk_index: int
    text: str
    estimated_tokens: int
    metadata: dict = field(default_factory=dict)


@dataclass
class AssembledContext:
    """The assembled context and its token-budget information."""

    entries: list[ContextEntry]
    rendered: str
    estimated_tokens: int
    max_tokens: int
    excluded_chunk_ids: list[str] = field(default_factory=list)


class ContextAssembler:
    """Assemble retrieved chunks within a configurable token budget."""

    def __init__(
        self,
        max_tokens: int = 4000,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        self.max_tokens = max_tokens
        self.token_counter = token_counter or CharacterTokenCounter()

    def assemble(self, retrieved_chunks: list[RetrievedChunk]) -> AssembledContext:
        """Select relevant chunks in retrieval order until the budget is reached."""

        entries: list[ContextEntry] = []
        excluded_chunk_ids: list[str] = []
        estimated_tokens = 0

        for chunk in retrieved_chunks:
            chunk_tokens = self.token_counter.count(chunk.text)

            if estimated_tokens + chunk_tokens > self.max_tokens:
                excluded_chunk_ids.append(chunk.chunk_id)
                continue

            citation_id = f"S{len(entries) + 1}"

            entries.append(
                ContextEntry(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    estimated_tokens=chunk_tokens,
                    metadata=dict(chunk.metadata or {}),
                )
            )

            estimated_tokens += chunk_tokens

        rendered = "\n\n".join(
            f"[{entry.citation_id}] Source: {entry.source}\n{entry.text}" for entry in entries
        )

        return AssembledContext(
            entries=entries,
            rendered=rendered,
            estimated_tokens=estimated_tokens,
            max_tokens=self.max_tokens,
            excluded_chunk_ids=excluded_chunk_ids,
        )
