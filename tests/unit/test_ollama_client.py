import json

import httpx
import pytest

from knowledge_assistant.generation.ollama_client import (
    DEFAULT_MODEL_NAME,
    OllamaClient,
    OllamaConfig,
)


def test_default_configuration():
    config = OllamaConfig()

    assert config.base_url == "http://localhost:11434"
    assert config.model == DEFAULT_MODEL_NAME
    assert config.timeout == 120.0


def test_custom_configuration():
    config = OllamaConfig(
        base_url="http://192.168.1.100:11434",
        model="llama3:latest",
        timeout=60.0,
    )

    assert config.base_url == "http://192.168.1.100:11434"
    assert config.model == "llama3:latest"
    assert config.timeout == 60.0


def test_generate_returns_response_text():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "mistral:7b-instruct-v0.3-q6_K",
                "response": "The inverter is operating normally.",
                "done": True,
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(http_client=http_client)

    response = client.generate("What is the inverter status?")

    assert response == "The inverter is operating normally."

    http_client.close()


def test_generate_sends_correct_request():
    captured_requests = []

    def handler(request):
        captured_requests.append(request)

        return httpx.Response(
            200,
            json={
                "response": "Generated answer",
                "done": True,
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(http_client=http_client)

    client.generate("Explain solar inverters")

    request = captured_requests[0]

    assert request.method == "POST"
    assert request.url.path == "/api/generate"

    payload = json.loads(request.content)

    assert payload["model"] == DEFAULT_MODEL_NAME
    assert payload["prompt"] == "Explain solar inverters"
    assert payload["stream"] is False

    http_client.close()


def test_generate_includes_system_prompt():
    captured_requests = []

    def handler(request):
        captured_requests.append(request)

        return httpx.Response(
            200,
            json={"response": "Generated answer", "done": True},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(http_client=http_client)

    client.generate(
        prompt="Explain this system",
        system="You are an industrial energy systems expert.",
    )

    payload = json.loads(captured_requests[0].content)

    assert payload["system"] == ("You are an industrial energy systems expert.")

    http_client.close()


def test_generate_passes_temperature():
    captured_requests = []

    def handler(request):
        captured_requests.append(request)

        return httpx.Response(
            200,
            json={"response": "Generated answer", "done": True},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(http_client=http_client)

    client.generate(
        prompt="Explain solar energy",
        temperature=0.5,
    )

    payload = json.loads(captured_requests[0].content)

    assert payload["options"]["temperature"] == 0.5

    http_client.close()


def test_generate_rejects_empty_prompt():
    client = OllamaClient()

    with pytest.raises(ValueError, match="prompt"):
        client.generate("")


def test_generate_handles_connection_error():
    def handler(request):
        raise httpx.ConnectError("Connection refused")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(http_client=http_client)

    with pytest.raises(ConnectionError):
        client.generate("Explain solar energy")

    http_client.close()


def test_generate_handles_http_error():
    def handler(request):
        return httpx.Response(
            500,
            json={"error": "Internal server error"},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(http_client=http_client)

    with pytest.raises(RuntimeError):
        client.generate("Explain solar energy")

    http_client.close()


@pytest.mark.parametrize(
    "response_data",
    [
        {},
        {"response": None},
        {"response": ""},
        {"response": 123},
    ],
)
def test_generate_rejects_malformed_response(response_data):
    def handler(request):
        return httpx.Response(200, json=response_data)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(http_client=http_client)

    with pytest.raises(RuntimeError):
        client.generate("Explain solar energy")

    http_client.close()


def test_close_closes_http_client():
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"response": "OK"}))
    )
    client = OllamaClient(http_client=http_client)

    client.close()

    assert http_client.is_closed


def test_context_manager_closes_http_client():
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"response": "OK"}))
    )

    with OllamaClient(http_client=http_client) as client:
        assert client.http_client.is_closed is False

    assert http_client.is_closed


def test_context_manager_closes_client_after_exception():
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"response": "OK"}))
    )

    with pytest.raises(RuntimeError):
        with OllamaClient(http_client=http_client):
            raise RuntimeError("Simulated failure")

    assert http_client.is_closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": ""},
        {"base_url": "   "},
        {"model": ""},
        {"model": "   "},
        {"timeout": 0},
        {"timeout": -1},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        OllamaConfig(**kwargs)


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1],
)
def test_generate_rejects_invalid_temperature(temperature):
    client = OllamaClient()

    with pytest.raises(ValueError, match="temperature"):
        client.generate(
            prompt="Explain solar energy",
            temperature=temperature,
        )

    client.close()
