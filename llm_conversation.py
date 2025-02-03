from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypedDict
import ollama
from rich.text import Text
from rich.console import Console
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter


@dataclass
class AIAgent:
    name: str
    system_prompt: str
    model: str
    temperature: float = 0.8
    ctx_size: int = 2048
    _messages: list[dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._messages = [{"role": "system", "content": self.system_prompt}]

    @property
    def messages(self) -> list[dict[str, str]]:
        return deepcopy(self._messages)

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def chat(self, user_input: str | None) -> str:
        # `None` user_input means the agent is starting the conversation or responding multiple times.
        if user_input is not None:
            self.add_message("user", user_input)

        response = ollama.chat(
            model=self.model,
            messages=self._messages,
            options={
                "num_ctx": self.ctx_size,
                "temperature": self.temperature,
            },
        )

        assistant_reply: str = response["message"]["content"]
        self.add_message("assistant", assistant_reply)
        return assistant_reply


@dataclass
class ConversationManager:
    class ConversationLogItem(TypedDict):
        agent: str
        content: str

    agent1: AIAgent
    agent2: AIAgent
    initial_message: str | None
    _conversation_log: list[ConversationLogItem] = field(
        default_factory=list, init=False
    )

    def save_conversation(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            _ = f.write(f"=== Agent 1 ===\n\n")
            _ = f.write(f"Name: {self.agent1.name}\n")
            _ = f.write(f"Model: {self.agent1.model}\n")
            _ = f.write(f"Temperature: {self.agent1.temperature}\n")
            _ = f.write(f"Context Size: {self.agent1.ctx_size}\n")
            _ = f.write(f"System Prompt: {self.agent1.system_prompt}\n\n")
            _ = f.write(f"=== Agent 2 ===\n\n")
            _ = f.write(f"Name: {self.agent2.name}\n")
            _ = f.write(f"Model: {self.agent2.model}\n")
            _ = f.write(f"Temperature: {self.agent2.temperature}\n")
            _ = f.write(f"Context Size: {self.agent2.ctx_size}\n")
            _ = f.write(f"System Prompt: {self.agent2.system_prompt}\n\n")
            _ = f.write(f"=== Conversation ===\n\n")

            for i, msg in enumerate(self._conversation_log):
                if i > 0:
                    _ = f.write("\n" + "\u2500" * 80 + "\n\n")

                _ = f.write(f"{msg['agent']}: {msg['content']}\n")

    def run_conversation(self):
        """
        Generate an iterator of conversation responses.

        :return: Iterator of (agent_name, message) tuples
        """

        last_message = self.initial_message
        is_agent1_turn = True

        # If a non-empty initial message is provided, start with it.
        if self.initial_message is not None:
            # Make the first agent the one to say the initial message, and the second agent the one to respond.
            self.agent1.add_message("assistant", self.initial_message)
            self._conversation_log.append(
                {"agent": self.agent1.name, "content": self.initial_message}
            )
            yield (self.agent1.name, self.initial_message)
            is_agent1_turn = False

        while True:
            current_agent = self.agent1 if is_agent1_turn else self.agent2
            last_message = current_agent.chat(last_message)
            self._conversation_log.append(
                {"agent": current_agent.name, "content": last_message}
            )
            yield (current_agent.name, last_message)
            is_agent1_turn = not is_agent1_turn


def get_available_models() -> list[str]:
    return [x.model or "" for x in ollama.list().models if x.model]


def create_ai_agent(console: Console, agent_number: int) -> AIAgent:
    console.print(f"=== Creating AI Agent {agent_number} ===", style="bold cyan")

    available_models = get_available_models()
    console.print("\nAvailable Models:", style="bold")
    for model in available_models:
        console.print(Text("• " + model))
    console.print("")

    while True:
        model_completer = WordCompleter(available_models, ignore_case=True)
        model_name = (
            prompt(
                f"Enter model name (default: {available_models[0]}): ",
                completer=model_completer,
                complete_while_typing=True,
            )
            or available_models[0]
        )

        if model_name in available_models:
            break

        console.print("Invalid model name!", style="bold red")

    while True:
        try:
            temperature_str: str = prompt("Enter temperature (default: 0.8): ") or "0.8"
            temperature: float = float(temperature_str)
            if not (0.0 <= temperature <= 1.0):
                raise ValueError("Temperature must be between 0.0 and 1.0")
            break
        except ValueError as e:
            console.print(f"Invalid input: {e}", style="bold red")

    while True:
        try:
            ctx_size_str: str = prompt("Enter context size (default: 2048): ") or "2048"
            ctx_size: int = int(ctx_size_str)
            if ctx_size < 0:
                raise ValueError("Context size must be a non-negative integer")
            break
        except ValueError as e:
            console.print(f"Invalid input: {e}", style="bold red")

    name = prompt(f"Enter name (default: AI {agent_number}): ") or f"AI {agent_number}"
    system_prompt = prompt(f"Enter system prompt for {name}: ")

    return AIAgent(
        name=name,
        system_prompt=system_prompt,
        model=model_name,
        temperature=temperature,
        ctx_size=ctx_size,
    )


def display_message(console: Console, agent_name: str, color: str, message: str):
    text = Text()
    _ = text.append(f"{agent_name}:", style=f"{color} bold")
    _ = text.append(f" {message}")
    console.print(text)


def main():
    color1: str = "blue"
    color2: str = "green"

    console = Console()
    console.clear()
    agent1 = create_ai_agent(console, 1)
    console.clear()
    agent2 = create_ai_agent(console, 2)
    console.clear()
    initial_message = prompt("Enter initial message (can be empty): ") or None
    console.clear()

    manager = ConversationManager(
        agent1=agent1, agent2=agent2, initial_message=initial_message
    )

    console.print("=== Conversation Started ===\n", style="bold cyan")
    is_first_message = True

    try:
        for agent_name, message in manager.run_conversation():
            if not is_first_message:
                console.print("")
                console.rule()
                console.print("")

            is_first_message = False
            color = color1 if agent_name == agent1.name else color2
            display_message(console, agent_name, color, message)

    except KeyboardInterrupt:
        console.print("\n=== Conversation Ended ===\n", style="bold cyan")
        manager.save_conversation("messages.txt")
        console.print("\nConversation saved to messages.txt\n\n", style="bold yellow")


if __name__ == "__main__":
    main()
