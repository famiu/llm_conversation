"""Module for the AIAgent class."""

from collections.abc import Iterator
from typing import cast

from openai import OpenAI
from pydantic import BaseModel


class AIAgent:
    """An AI agent for conversational AI using OpenAI-compatible providers."""

    name: str
    model: str
    temperature: float = 0.8
    ctx_size: int = 2048
    client: OpenAI
    # TODO: Use a memory system instead to not grow context size indefinitely.
    _messages: list[dict[str, str]]

    def __init__(
        self,
        name: str,
        model: str,
        temperature: float,
        ctx_size: int,
        system_prompt: str,
        client: OpenAI,
    ) -> None:
        """Initialize an AI agent.

        Args:
            name (str): Name of the AI agent
            model (str): Model to be used
            temperature (float): Sampling temperature for the model (0.0-1.0)
            ctx_size (int): Context size for the model
            system_prompt (str): Initial system prompt for the agent
            provider (ProviderConfig): Provider configuration for API access
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        self.ctx_size = ctx_size
        self.client = client
        self._messages = [{"role": "system", "content": system_prompt}]

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return self._messages[0]["content"]

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set the system prompt for the agent."""
        self._messages[0]["content"] = value

    def add_message(self, name: str, role: str, content: str) -> None:
        """Add a message to the end of the conversation history."""
        self._messages.append({"name": name, "role": role, "content": content})

    def get_response(self, output_format: type[BaseModel]) -> Iterator[str]:
        """Generate a response message based on the conversation history.

        Args:
            output_format (type[BaseModel]): Pydantic model for structured output format

        Yields:
            str: Chunk of the response from the agent

        Raises:
            RuntimeError: If the provider does not support structured output (SSR)
        """
        response_stream = self.client.chat.completions.create(  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
            model=self.model,
            messages=self._messages,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
            temperature=self.temperature,
            max_tokens=self.ctx_size,
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_format.__name__,
                    "schema": output_format.model_json_schema(),
                    "strict": True,
                },
            },  # type: ignore[typeddict-item]
        )

        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                content: str = chunk.choices[0].delta.content
                yield content  # Stream JSON chunks as they arrive

    def get_param_count(self) -> int:
        """Get the number of parameters in the model (when supported by provider)."""
        try:
            # Try to get model info - most providers don't expose parameter count
            model = self.client.models.retrieve(self.model)
            return cast(int, getattr(model, "parameter_count", 0))
        except Exception:
            return 0  # Fallback when not supported
