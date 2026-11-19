from dataclasses import dataclass

import httpx

DEFAULT_MODEL_NAME = "mistral:7b-instruct-v0.3-q6_K"


@dataclass
class OllamaConfig:
    """Configuration for the Ollama API client."""

    base_url: str = "http://localhost:11434"
    model: str = DEFAULT_MODEL_NAME
    timeout: float = 120.0

    def __post_init__(self):
        if not self.base_url.strip():
            raise ValueError("base_url cannot be empty")

        if not self.model.strip():
            raise ValueError("model cannot be empty")

        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")


class OllamaClient:
    """HTTP client for generating text with Ollama.

    The client owns its HTTP connection and closes it when its
    lifecycle ends, including when an HTTP client is injected.
    """

    def __init__(
        self,
        config: OllamaConfig | None = None,
        http_client: httpx.Client | None = None,
    ):
        self.config = config or OllamaConfig()

        self.http_client = http_client or httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text using the configured Ollama model."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system is not None:
            payload["system"] = system

        try:
            response = self.http_client.post(
                f"{self.config.base_url.rstrip('/')}/api/generate",
                json=payload,
            )
            response.raise_for_status()

        except httpx.ConnectError as exc:
            raise ConnectionError(f"Could not connect to Ollama at {self.config.base_url}") from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}") from exc

        except httpx.RequestError as exc:
            raise ConnectionError(f"Ollama request failed: {exc}") from exc

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc

        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned an invalid response")

        generated_text = result.get("response")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError("Ollama response does not contain valid text")

        return generated_text
