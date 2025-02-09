"""Module for the AIAgent class."""

from collections.abc import Iterator
from typing import Any, cast

import ollama
from pydantic import BaseModel


class AIAgent:
    """An AI agent for conversational AI using Ollama models."""

    name: str
    model: str
    temperature: float = 0.8
    ctx_size: int = 2048
    _messages: list[dict[str, str]]

    def __init__(
        self,
        name: str,
        model: str,
        temperature: float,
        ctx_size: int,
        system_prompt: str,
    ) -> None:
        """Initialize an AI agent.

        Args:
            name (str): Name of the AI agent
            model (str): Ollama model to be used
            temperature (float): Sampling temperature for the model (0.0-1.0)
            ctx_size (int): Context size for the model
            system_prompt (str): Initial system prompt for the agent
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        self.ctx_size = ctx_size
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
            user_input (str | None): User input to the agent

        Yields:
            str: Chunk of the response from the agent
        """
        response_stream = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
            model=self.model,
            messages=self._messages,
            options={
                "num_ctx": self.ctx_size,
                "temperature": self.temperature,
            },
            stream=True,
            format=output_format.model_json_schema(),
        )

        chunks: list[str] = []
        for chunk in response_stream:
            content: str = chunk["message"]["content"]
            chunks.append(content)
            yield content  # Stream chunks as they arrive

    def get_param_count(self) -> int:
        """Get the number of parameters in the model."""
        return cast(int, cast(dict[str, Any], ollama.show(self.model).modelinfo)["general.parameter_count"])
