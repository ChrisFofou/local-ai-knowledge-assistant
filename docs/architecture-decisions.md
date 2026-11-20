# Architecture Decision Records

This document records important architectural and technical decisions made
during development of the Local AI Knowledge Assistant.

Each decision documents the problem, alternatives, chosen approach,
rationale, trade-offs, and validation strategy.

---

## ADR-005: Context Assembly and Token Budget Management

**Status:** Accepted
**Date:** 2026-11-19

### Context

The semantic retrieval component returns ranked document chunks. Before these
chunks are passed to an LLM, the system needs to assemble relevant context
while respecting the model's context-window constraints.

Context assembly must preserve document provenance so generated answers can
reference their original sources.

Retrieval count and context budget are separate controls. Retrieving a
certain number of chunks does not guarantee that all retrieved content will
fit into the prompt.

### Decision

Implement context assembly as a separate component, independent of retrieval
and LLM generation.

The context assembler will:

- Accept retrieved chunks in relevance-ranked order.
- Preserve chunk content and source metadata.
- Assign sequential citation identifiers (S1, S2, S3, etc.).
- Enforce a configurable context token budget.
- Include chunks when they fit within the remaining budget.
- Skip chunks that do not fit and continue evaluating later chunks.
- Never truncate chunks in the initial implementation.
- Return structured context entries alongside a rendered context string.
- Support an injectable token-counting strategy.

The initial context budget is **4,000 estimated tokens**.

The initial token estimator uses the heuristic:

    estimated_tokens = ceil(character_count / 3)

This is a rough approximation, not an actual model tokenizer.

### Alternatives Considered

#### 1. Fixed number of retrieved chunks

Always include a fixed number of chunks, for example five.

Advantages:
- Very simple.
- Easy to configure.

Disadvantages:
- Chunk lengths vary.
- A fixed count does not control prompt size.
- Long chunks may consume a disproportionate amount of the model context.

Rejected because it does not reliably manage context-window constraints.

#### 2. Character-based estimation at four characters per token

Advantages:
- Simple.
- Common rough approximation for English text.

Disadvantages:
- Can underestimate token usage.
- Tokenization varies by language, vocabulary, punctuation, and model.

Rejected as the initial budgeting heuristic in favor of a more conservative three-character ratio.

#### 3. Word-count-based estimation

Advantages:
- Simple.
- Can approximate token count for ordinary prose.

Disadvantages:
- Weak for technical identifiers, code, punctuation, numbers, and multilingual text.
- Languages without whitespace-delimited words are problematic.

Rejected because character-based estimation is simpler and more language-independent at the implementation level.

#### 4. Model-specific tokenizer

Advantages:
- Much more accurate token counting.
- Better budget utilization.
- Can reflect the actual target model's tokenization.

Disadvantages:
- Requires tokenizer dependency and model compatibility.
- Adds complexity.
- Tokenizer selection becomes coupled to the generation model unless carefully abstracted.

Deferred. The token counter will be replaceable so a model-specific tokenizer can be added without redesigning context assembly.

#### 5. Truncate chunks to fit

Advantages:
- Can use nearly all available context budget.
- Prevents one chunk from exceeding the remaining space.

Disadvantages:
- May cut off sentences or important information.
- Can damage semantic coherence.
- Makes source evidence harder to interpret.

Rejected initially to preserve chunk integrity and citation traceability.

#### 6. Global optimization of chunk selection

Algorithms such as knapsack optimization could select a combination of chunks that maximizes relevance under the token budget.

Advantages:
- Could maximize total relevance within a budget.

Disadvantages:
- Adds algorithmic complexity.
- Retrieved chunks are already relevance-ranked.
- Relevance scores may not be calibrated as utility values.
- More complex selection may disrupt retrieval order.

Rejected for the initial version in favor of greedy relevance-ranked selection.

### Algorithm

Use greedy selection in retrieval order:

1. Initialize remaining budget to the configured maximum.
2. Process chunks in retrieval order.
3. Estimate the cost of each chunk, including its citation and source formatting.
4. If the chunk fits, include it and reduce the remaining budget.
5. If it does not fit, exclude it and continue to the next chunk.
6. Assign citation IDs sequentially to included chunks.
7. Return included entries, excluded chunk IDs, rendered context, and token accounting.

The selection algorithm is O(n) in the number of retrieved chunks, excluding token-counting costs.

### Trade-offs

Benefits:
- Simple and predictable.
- Preserves retrieval relevance order.
- Avoids truncating source content.
- Allows smaller chunks to be included even when a larger chunk does not fit.
- Easy to unit test.
- Token counter can be replaced independently.

Limitations:
- Character-based estimation is approximate.
- The selected subset is not globally optimal.
- Skipping a chunk can omit important context.
- The initial budgeting method may be inaccurate for some languages and technical content.

### Validation Strategy

Unit tests will verify:

- Empty input.
- Single and multiple chunks.
- Retrieval-order preservation.
- Citation numbering.
- Source and metadata preservation.
- Token estimation.
- Context-budget enforcement.
- Oversized-chunk skipping.
- Excluded-chunk tracking.
- Custom token-counter injection.
- Invalid configuration rejection.
- Rendered context format.

Integration testing will later verify context assembly with real retrieved chunks.

### Future Improvements

- Replace heuristic token estimation with a tokenizer compatible with the selected LLM.
- Evaluate chunk-selection strategies against retrieval quality and answer quality.
- Consider relevance-aware optimization if evaluation demonstrates a benefit.
- Explore context deduplication for overlapping chunks.
- Measure actual prompt-token usage against estimated usage.
