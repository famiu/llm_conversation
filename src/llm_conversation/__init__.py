"""Main module for LLM Conversation package."""

import argparse
import typing
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path

import distinctipy  # type: ignore[import-untyped] # pyright: ignore[reportMissingTypeStubs]
from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter, WordCompleter
from prompt_toolkit.validation import Validator
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from .ai_agent import AIAgent
from .config import (
    DEFAULT_PROVIDERS,
    PROVIDER_NAMES_CAPITALIZED,
    AgentConfig,
    Config,
    ProviderConfig,
    TurnOrder,
    create_openai_client,
    get_available_models,
    load_config,
    validate_provider_config,
)
from .conversation_manager import ConversationManager


def create_ai_agent_from_input(
    console: Console, placeholder_name: str, provider_api_keys: dict[str, str | None]
) -> AIAgent:
    """Create an AIAgent instance from user input.

    Args:
        console (Console): Rich console instance.
        placeholder_name (str): Default name for the agent.

    Returns:
        AIAgent: Created AI agent instance.
    """
    while True:
        console.clear()
        console.print(f'=== Creating AI Agent: "{placeholder_name}" ===\n', style="bold cyan")

        # Step 1: Choose provider
        available_providers = list(DEFAULT_PROVIDERS.keys()) + ["custom"]
        console.print("Available Providers:", style="bold")
        for provider_name in available_providers:
            console.print(Text("• " + provider_name))
        console.print("")

        provider_name = (
            prompt(
                "Enter provider name (default: ollama): ",
                completer=FuzzyWordCompleter(available_providers),
                complete_while_typing=True,
                validator=Validator.from_callable(
                    lambda text: text == "" or text in available_providers,
                    error_message="Provider not found",
                    move_cursor_to_end=True,
                ),
                validate_while_typing=False,
            )
            or "ollama"
        )

        # Step 2: Configure provider
        provider_name_capitalized = (
            PROVIDER_NAMES_CAPITALIZED.get(provider_name, provider_name)
            if provider_name in DEFAULT_PROVIDERS
            else "Custom"
        )
        cache_key: str = provider_name.lower()
        provider: ProviderConfig | None = None

        console.print(f"\n=== {provider_name_capitalized} Configuration ===\n", style="bold cyan")

        if provider_name == "custom":
            base_url = prompt("Enter base URL (e.g., https://api.example.com/v1): ")
            # Ensure that custom providers with different base URLs are cached separately.
            cache_key += f":{base_url}"
            provider = ProviderConfig(
                base_url=base_url,
                api_key=None,  # Custom providers may not require an API key
            )
        else:
            # All non-custom providers have a default base URL.
            # Unlike the config file, the interactive prompt does not allow changing the base URL.
            provider = DEFAULT_PROVIDERS[provider_name].model_copy()

        assert provider.api_key is None, "Provider API key should be None initially"

        if provider_name in provider_api_keys:
            provider.api_key = provider_api_keys[provider_name]
            console.print(f"ℹ️ Using stored API key for {provider_name_capitalized}.", style="bold blue")
        else:
            provider.api_key = (
                prompt(f"Enter API key for {provider_name_capitalized} (optional): ", is_password=True).strip() or None
            )

            console.print("\nValidating API key...", style="yellow")

            if not validate_provider_config(provider):
                console.print("❌ Invalid API key or provider unreachable!", style="bold red")
                console.print("Please try again with a valid configuration.\n", style="yellow")
                _ = prompt("Press Enter to retry or Ctrl+C to exit...")
                continue  # Retry

            console.print("✅ API key is valid!", style="bold green")
            console.print("ℹ️ This key will be remembered for this session.", style="bold blue")
            # Store the validated API key in the session cache
            provider_api_keys[provider_name] = provider.api_key

        # Step 3: Get available models
        console.print(f"\nFetching available models from {provider_name_capitalized}...", style="yellow")
        available_models = get_available_models(provider)

        if not available_models:
            console.print(f"❌ No models available from {provider_name_capitalized}!", style="bold red")
            console.print("Please try a different provider or check your configuration.\n", style="yellow")
            _ = prompt("Press Enter to retry or Ctrl+C to exit...")
            continue  # Retry

        console.print("Available Models:", style="bold")
        for model in available_models[:20]:  # Limit display for readability
            console.print(Text("• " + model))
        if len(available_models) > 20:
            console.print(Text(f"... and {len(available_models) - 20} more"))
        console.print("")

        # Step 4: Configure model and agent settings
        model_name = prompt(
            "Enter model name: ",
            completer=FuzzyWordCompleter(available_models, WORD=True),
            complete_while_typing=True,
            validator=Validator.from_callable(
                lambda text: text == "" or text in available_models,
                error_message="Model not found",
                move_cursor_to_end=True,
            ),
            validate_while_typing=False,
        )

        def _validate_float(text: str) -> bool:
            if text == "":
                return True
            try:
                _ = float(text)
            except ValueError:
                return False
            return True

        temperature_str: str = (
            prompt(
                "Enter temperature (default: 0.8): ",
                validator=Validator.from_callable(
                    lambda text: text == "" or _validate_float(text) and 0.0 <= float(text) <= 1.0,
                    error_message="Temperature must be a number between 0.0 and 1.0",
                    move_cursor_to_end=True,
                ),
            )
            or "0.8"
        )
        temperature: float = float(temperature_str)

        ctx_size_str: str = (
            prompt(
                "Enter context size (default: 2048): ",
                validator=Validator.from_callable(
                    lambda text: text == "" or text.isdigit() and int(text) >= 0,
                    error_message="Context size must be a non-negative integer",
                    move_cursor_to_end=True,
                ),
            )
            or "2048"
        )
        ctx_size: int = int(ctx_size_str)

        name = prompt(f"Enter name (default: {placeholder_name}): ") or placeholder_name
        system_prompt = prompt(f"Enter system prompt for {name}: ")

        agent = AIAgent(
            name=name,
            model=model_name,
            temperature=temperature,
            ctx_size=ctx_size,
            system_prompt=system_prompt,
            client=create_openai_client(provider),
        )

        return agent


def create_ai_agent_from_config(agent_config: AgentConfig, config: Config) -> AIAgent:
    """Create an AIAgent instance from configuration.

    Args:
        agent_config (AgentConfig): Configuration for the agent.
        config (Config): Full configuration containing provider details.

    Returns:
        AIAgent: Created AI agent instance.
    """
    provider = config.providers[agent_config.provider]

    return AIAgent(
        name=agent_config.name,
        model=agent_config.model,
        temperature=agent_config.temperature,
        ctx_size=agent_config.ctx_size,
        system_prompt=agent_config.system_prompt,
        client=create_openai_client(provider),
    )


def markdown_to_text(markdown_content: str) -> Text:
    """Convert Markdown content to a styled Text object."""
    console = Console()
    md = Markdown(markdown_content)
    segments = list(console.render(md))
    result = Text()
    for segment in segments:
        _ = result.append(segment.text, style=segment.style)

    result.rstrip()
    return result


def display_message(
    console: Console,
    agent_name: str,
    name_color: tuple[int, int, int],
    message_stream: Iterator[str],
    use_markdown: bool = False,
) -> None:
    """Display a message from an agent in the console.

    Args:
        console (Console): Rich console instance.
        agent_name (str): Name of the agent.
        name_color (str): Color to use for the agent name.
        message_stream (Iterator[str]): Stream of the entire message up until the newest chunk received.
        use_markdown (bool, optional): Whether to use Markdown for text formatting. Defaults to False.
    """
    # Create the agent name prefix as a Text object.
    agent_prefix = Text(f"{agent_name}: ", style=f"rgb({name_color[0]},{name_color[1]},{name_color[2]})")

    with Live("", console=console, transient=False) as live:
        for message in message_stream:
            # If the message is in Markdown format, convert it to a styled Text object, so we can append the
            # agent_prefix to it and display it through live.update().
            message_text = markdown_to_text(message) if use_markdown else Text(message)
            message_text.style = "default"
            live.update(agent_prefix + message_text, refresh=True)


def prompt_bool(prompt_text: str, default: bool = False) -> bool:
    """Prompt the user with a yes/no question and return the response as a boolean.

    Args:
        prompt_text (str): Prompt text to display.
        default (bool, optional): Default value to return if the user input is invalid. Defaults to False.

    Returns:
        bool: True if the user input is "yes" or "y" (case-insensitive), False otherwise.
    """
    response = prompt(prompt_text).lower()

    if not response or response not in ["y", "yes", "n", "no"]:
        return default

    return response[0] == "y"


# TODO: Add a GUI.
# TODO: Add a logging system for debugging.
def main() -> None:
    """Run a conversation between AI agents."""
    parser = argparse.ArgumentParser(description="Run a conversation between AI agents")
    _ = parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {version('llm-conversation')}")
    _ = parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Save conversation log to file. Uses JSON format if file extension is .json, "
        + "otherwise uses plain text format",
    )
    _ = parser.add_argument("-c", "--config", type=Path, help="Use JSON configuration file")
    args = parser.parse_args()

    console = Console()
    console.clear()

    # TODO: Remove truecolor requirement. Either replace distinctipy with a library that supports 8-bit colors, or
    #       implement a custom color mapping that converts distinctipy's RGB colors to 8-bit ANSI colors.
    if console.color_system != "truecolor":
        console.print("Please run this program in a terminal with true color support.", style="bold red")
        return

    agents: list[AIAgent]
    turn_order: TurnOrder
    moderator: AIAgent | None = None

    if args.config:
        # Load from config file
        config = load_config(args.config)
        agents = [create_ai_agent_from_config(agent_config, config) for agent_config in config.agents]
        settings = config.settings
        use_markdown = settings.use_markdown or False
        allow_termination = settings.allow_termination or False
        initial_message = settings.initial_message
        turn_order = settings.turn_order
        moderator = create_ai_agent_from_config(settings.moderator, config) if settings.moderator else None
    else:
        agent_count_str: str = (
            prompt(
                "Enter the number of AI agents (default: 2): ",
                validator=Validator.from_callable(
                    lambda text: text == "" or text.isdigit() and 1 < int(text) <= 10,
                    error_message="Number of agents must be an integer greater than 1 and not more than 10",
                    move_cursor_to_end=True,
                ),
            )
            or "2"
        )

        agent_count: int = int(agent_count_str)
        agents = []
        provider_api_keys: dict[str, str | None] = {}  # Session cache for API keys

        for i in range(agent_count):
            agent = create_ai_agent_from_input(console, f"AI {i + 1}", provider_api_keys)
            agents.append(agent)
        console.clear()

        use_markdown = prompt_bool("Use Markdown for text formatting? (y/N): ", default=False)
        allow_termination = prompt_bool("Allow AI agents to terminate the conversation? (y/N): ", default=False)
        initial_message = prompt("Enter initial message (can be empty): ") or None

        turn_order_values = typing.cast(list[str], list(typing.get_args(TurnOrder)))
        turn_order = typing.cast(
            TurnOrder,
            prompt(
                "Enter turn order (default: round_robin): ",
                completer=WordCompleter(turn_order_values, ignore_case=True),
                validator=Validator.from_callable(
                    lambda text: text == "" or text in turn_order_values,
                    error_message="Invalid turn order",
                    move_cursor_to_end=True,
                ),
            )
            or "round_robin",
        )

        if turn_order == "moderator" and prompt_bool("Configure the moderator agent? (y/N): ", default=False):
            moderator = create_ai_agent_from_input(console, "Moderator", provider_api_keys)
        console.clear()

    manager = ConversationManager(
        agents=agents,
        initial_message=initial_message,
        use_markdown=use_markdown,
        allow_termination=allow_termination,
        turn_order=turn_order,
        moderator=moderator,
    )

    # Get distinct colors for each agent. distinctipy.get_colors() returns floats between 0 and 1, so convert to 0-255
    # by multiplying by 255. This is necessary because Rich expects color values in the 0-255 range.
    colors = distinctipy.get_colors(len(agents), pastel_factor=0.6)  # pyright: ignore[reportUnknownMemberType]
    agent_name_color: dict[str, tuple[int, int, int]] = {
        agent.name: (int(r * 255), int(g * 255), int(b * 255)) for agent, (r, g, b) in zip(agents, colors)
    }

    console.print("=== Conversation Started ===\n", style="bold cyan")
    is_first_message = True

    try:
        for agent_name, message in manager.run_conversation():
            if not is_first_message:
                console.print("")
                console.rule()
                console.print("")

            is_first_message = False
            display_message(console, agent_name, agent_name_color[agent_name], message, use_markdown)

        console.print("\n=== Conversation Terminated by Agent ===\n", style="bold cyan")

    except KeyboardInterrupt:
        console.print("\n=== Conversation Terminated by User ===\n", style="bold cyan")

    if args.output is not None:
        manager.save_conversation(args.output)
        console.print(f"\nConversation saved to {args.output}\n\n", style="bold yellow")


if __name__ == "__main__":
    main()
