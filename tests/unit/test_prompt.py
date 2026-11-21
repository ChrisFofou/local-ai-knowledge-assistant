import pytest

from knowledge_assistant.rag.context import AssembledContext, ContextEntry
from knowledge_assistant.rag.prompt import Prompt, PromptBuilder


@pytest.fixture
def assembled_context():
    entry = ContextEntry(
        citation_id="S1",
        chunk_id="chunk-1",
        document_id="doc-1",
        source="solar_guide.pdf",
        chunk_index=0,
        text="Solar inverters convert DC into AC.",
        estimated_tokens=12,
        metadata={"language": "en"},
    )

    return AssembledContext(
        entries=[entry],
        rendered="[S1] Source: solar_guide.pdf\nSolar inverters convert DC into AC.",
        estimated_tokens=12,
        max_tokens=4000,
        excluded_chunk_ids=[],
    )


def test_build_returns_prompt_object(assembled_context):
    builder = PromptBuilder()

    result = builder.build(
        question="What do solar inverters do?",
        context=assembled_context,
    )

    assert isinstance(result, Prompt)
    assert result.system
    assert result.user


def test_system_prompt_requires_context_based_answers(assembled_context):
    builder = PromptBuilder()
    result = builder.build("What do solar inverters do?", assembled_context)

    assert "provided context" in result.system.lower()


def test_system_prompt_requires_citations(assembled_context):
    builder = PromptBuilder()
    result = builder.build("What do solar inverters do?", assembled_context)

    assert "citation" in result.system.lower()
    assert "[S1]" in result.system


def test_system_prompt_prevents_hallucinations(assembled_context):
    builder = PromptBuilder()
    result = builder.build("What do solar inverters do?", assembled_context)

    assert "do not invent" in result.system.lower()


def test_system_prompt_treats_context_as_untrusted_data(assembled_context):
    builder = PromptBuilder()
    result = builder.build("What do solar inverters do?", assembled_context)

    assert "instructions" in result.system.lower()


def test_user_prompt_contains_question_and_context(assembled_context):
    builder = PromptBuilder()
    result = builder.build("What do solar inverters do?", assembled_context)

    assert "What do solar inverters do?" in result.user
    assert "Solar inverters convert DC into AC." in result.user
    assert "[S1]" in result.user


def test_user_prompt_preserves_source_information(assembled_context):
    builder = PromptBuilder()
    result = builder.build("What do solar inverters do?", assembled_context)

    assert "solar_guide.pdf" in result.user


def test_empty_context_is_handled():
    context = AssembledContext(
        entries=[],
        rendered="",
        estimated_tokens=0,
        max_tokens=4000,
        excluded_chunk_ids=[],
    )

    result = PromptBuilder().build("What is a solar inverter?", context)

    assert "What is a solar inverter?" in result.user
    assert "no relevant context" in result.user.lower()


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_build_rejects_empty_question(question, assembled_context):
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="question"):
        builder.build(question, assembled_context)


def test_build_strips_surrounding_question_whitespace(assembled_context):
    result = PromptBuilder().build(
        "  What do solar inverters do?  ",
        assembled_context,
    )

    assert "What do solar inverters do?" in result.user
    assert "  What do solar inverters do?  " not in result.user
