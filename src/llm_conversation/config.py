"""Configuration loading and validation for the AI agents and conversation settings.

This module defines Pydantic models for the AI agents and conversation settings, and allows loading and validating the
configuration from a JSON file.
"""

import json
from pathlib import Path
from typing import Literal, Self

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

TurnOrder = Literal["round_robin", "random", "chain", "moderator", "vote"]


class ProviderConfig(BaseModel):
    """Configuration for any OpenAI-compatible provider."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    base_url: str | None = Field(default=None, description="Base URL for the provider API")
    api_key: str | None = Field(default=None, description="API key for the provider")


# Built-in provider definitions
DEFAULT_PROVIDERS = {
    "ollama": ProviderConfig(base_url="http://localhost:11434/v1", api_key=None),
    "openai": ProviderConfig(base_url="https://api.openai.com/v1", api_key=None),
    "anthropic": ProviderConfig(base_url="https://api.anthropic.com/v1", api_key=None),
    "google": ProviderConfig(base_url="https://generativelanguage.googleapis.com/v1beta/openai", api_key=None),
    "openrouter": ProviderConfig(base_url="https://openrouter.ai/api/v1", api_key=None),
    "together": ProviderConfig(base_url="https://api.together.xyz/v1", api_key=None),
    "groq": ProviderConfig(base_url="https://api.groq.com/openai/v1", api_key=None),
    "deepseek": ProviderConfig(base_url="https://api.deepseek.com/v1", api_key=None),
}

# Capitalized names for providers, used in UI or display contexts.
PROVIDER_NAMES_CAPITALIZED = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "openrouter": "OpenRouter",
    "together": "Together",
    "groq": "Groq",
    "deepseek": "DeepSeek",
}


def create_openai_client(provider: ProviderConfig) -> OpenAI:
    """Create an OpenAI client with the given provider configuration.

    Uses a placeholder API key if none is provided.

    Args:
        provider: Provider configuration containing base_url and api_key

    Returns:
        OpenAI: Configured OpenAI client instance
    """
    return OpenAI(base_url=provider.base_url, api_key=provider.api_key or "dummy-key")


def get_available_models(provider: ProviderConfig) -> list[str]:
    """Get a list of available models from OpenAI-compatible provider."""
    try:
        client = create_openai_client(provider)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception:
        return []  # Graceful fallback


def validate_provider_config(provider: ProviderConfig) -> bool:
    """Validate provider configuration by testing API key without consuming credits.

    Args:
        provider: Provider configuration to validate

    Returns:
        bool: True if valid, False if invalid
    """
    try:
        client = create_openai_client(provider)
        # Use models.list() endpoint - doesn't consume credits
        _ = client.models.list()
        return True
    except Exception:
        return False


class AgentConfig(BaseModel):
    """Configuration for an AI agent."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    name: str = Field(..., min_length=1, description="Name of the AI agent")
    model: str = Field(..., description="Model to be used")
    system_prompt: str = Field(..., description="Initial system prompt for the agent")
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for the model (0.0-1.0)",
    )
    ctx_size: int = Field(default=2048, ge=0, description="Context size for the model")
    provider: str = Field(default="ollama", description="Provider to use for this agent")


class ConversationSettings(BaseModel):
    """Extra settings for the conversation, not specific to any AI agent."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    use_markdown: bool = Field(default=False, description="Enable Markdown formatting")
    # TODO: Make termination make the agent leave the conversation instead of ending it.
    #       Only end the conversation if all agents have left.
    allow_termination: bool = Field(default=False, description="Allow AI agents to terminate the conversation")
    initial_message: str | None = Field(default=None, description="Initial message to start the conversation")
    # TODO: Add a turn order that lets conversations feel more natural instead of systematic.
    turn_order: TurnOrder = Field(default="round_robin", description="Strategy for selecting the next agent")
    moderator: AgentConfig | None = Field(
        default=None, description='Configuration for the moderator agent (if using "moderator" turn order)'
    )

    @model_validator(mode="after")
    def validate_moderator(self) -> Self:  # noqa: D102
        if self.turn_order != "moderator" and self.moderator is not None:
            raise ValueError("moderator can only be defined when turn_order is 'moderator'")

        return self


class Config(BaseModel):
    """Configuration for the AI agents and conversation settings."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    providers: dict[str, ProviderConfig] = Field(default_factory=dict, description="Provider configurations")
    agents: list[AgentConfig] = Field(..., description="Configuration for AI agents")
    settings: ConversationSettings = Field(..., description="Conversation settings")

    @model_validator(mode="after")
    def validate_and_merge_providers(self) -> Self:  # noqa: D102
        # Start with built-in providers
        merged_providers = DEFAULT_PROVIDERS.copy()

        # Override with user-defined providers
        for name, config in self.providers.items():
            if name in merged_providers:
                # Merge: use user values, keep defaults for None fields
                default = merged_providers[name]
                merged_providers[name] = ProviderConfig(
                    base_url=config.base_url or default.base_url, api_key=config.api_key or default.api_key
                )
            else:
                # New provider - must have base_url
                if not config.base_url:
                    msg = f"Custom provider '{name}' must specify base_url"
                    raise ValueError(msg)
                merged_providers[name] = config

        self.providers = merged_providers

        # Validate all agent provider references exist
        for agent in self.agents:
            if agent.provider not in self.providers:
                msg = f"Agent '{agent.name}' references unknown provider: {agent.provider}"
                raise ValueError(msg)

        if self.settings.moderator and self.settings.moderator.provider not in self.providers:
            msg = f"Moderator references unknown provider: {self.settings.moderator.provider}"
            raise ValueError(msg)

        return self


def load_config(config_path: Path) -> Config:
    """Load and validate the configuration file using Pydantic.

    Args:
        config_path (Path): Path to the JSON configuration file

    Returns:
        Config: Validated configuration object

    Raises:
        ValueError: If the configuration is invalid
    """
    try:
        with open(config_path) as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in config file: {e}"
        raise ValueError(msg)

    try:
        return Config.model_validate(config_dict)
    except Exception as e:
        msg = f"Configuration validation failed: {e}"
        raise ValueError(msg)
