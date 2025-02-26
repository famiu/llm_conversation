"""Module for managing a conversation between AI agents."""

import enum
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from partial_json_parser import ensure_json  # type: ignore[import-untyped] # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, Field, create_model

from .ai_agent import AIAgent

TurnOrder = Literal["round_robin", "random", "chain", "moderator", "vote"]


@dataclass
class ConversationManager:
    """Manager for a conversation between AI agents."""

    class _ConversationLogItem(TypedDict):
        agent: str
        content: str

    agents: list[AIAgent]
    initial_message: str | None
    use_markdown: bool = False
    allow_termination: bool = False
    turn_order: TurnOrder = "round_robin"
    moderator: AIAgent | None = None
    _conversation_log: list[_ConversationLogItem] = field(default_factory=list, init=False)
    _original_system_prompts: list[str] = field(init=False, repr=False)
    _output_format: type[BaseModel] = field(init=False, repr=False)
    _agent_name_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D105
        self._agent_name_to_idx = {}

        for i, agent in enumerate(self.agents):
            if agent.name in self._agent_name_to_idx:
                raise ValueError(f"Agent names must be unique: {agent.name}")
            self._agent_name_to_idx[agent.name] = i

        self._original_system_prompts = [agent.system_prompt for agent in self.agents]

        output_format_kwargs: dict[str, Any] = {
            "message": (str, Field(description="Message content")),
        }

        if self.allow_termination:
            output_format_kwargs["terminate"] = (
                bool,
                Field(
                    title="Terminate",
                    description="Terminate conversation if you believe it has reached a natural conclusion. "
                    + "Do not set this field to `true` unless you are certain the conversation should end.",
                ),
            )

        self._output_format = create_model("OutputFormat", **output_format_kwargs)

        # Modify system prompt to include termination instructions if allowed
        additional_instructions: str = ""

        if self.use_markdown:
            additional_instructions += (
                "You may use Markdown for text formatting. "
                "Examples: *italic*, **bold**, `code`, [link](https://example.com), etc.\n\n"
            )

        if self.allow_termination:
            additional_instructions += (
                "If you believe the conversation has reached a natural conclusion, you may choose to end the "
                + "conversation by setting the `terminate` field to `true` in your response. Only do this if you are "
                + "certain the conversation should end. When you end the conversation, also provide an in-character "
                + "final message to conclude the conversation.\n\n"
            )

        # Updated system prompts for each agent to give the agents more context about the conversation and their role.
        for agent in self.agents:
            other_agents = ", ".join([a.name for a in self.agents if a != agent])
            agent.system_prompt = (
                f"You are named {agent.name}. The other characters are {other_agents}. "
                + "Your task is to play the role you're given and continue the conversation.\n\n"
                + f"This is the prompt for your role: {agent.system_prompt}\n\n"
                + additional_instructions
            )

        # If the turn order is set to "moderator" and a moderator agent is not provided, create one.
        if self.turn_order == "moderator" and self.moderator is None:
            self._create_moderator_agent()
        elif self.turn_order != "moderator" and self.moderator is not None:
            raise ValueError("Cannot use a moderator agent without the turn order set to 'moderator'")

    def save_conversation(self, filename: Path) -> None:
        """Save the conversation log to a file.

        Args:
            filename (Path): Path to save the conversation log to
        """
        with open(filename, "w", encoding="utf-8") as f:
            if filename.suffix == ".json":

                def agent_to_dict(agent_idx: int) -> dict[str, Any]:
                    agent = self.agents[agent_idx]

                    return {
                        "name": agent.name,
                        "model": agent.model,
                        "temperature": agent.temperature,
                        "ctx_size": agent.ctx_size,
                        "system_prompt": self._original_system_prompts[agent_idx],
                    }

                # Save conversation log as JSON
                json.dump(
                    {
                        "agents": [agent_to_dict(i) for i in range(len(self.agents))],
                        "conversation": self._conversation_log,
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
                return

            # Save conversation log as plain text
            for i, agent in enumerate(self.agents, start=1):
                _ = f.write(f"=== Agent {i} ===\n\n")
                _ = f.write(f"Name: {agent.name}\n")
                _ = f.write(f"Model: {agent.model}\n")
                _ = f.write(f"Temperature: {agent.temperature}\n")
                _ = f.write(f"Context Size: {agent.ctx_size}\n")
                _ = f.write(f"System Prompt: {self._original_system_prompts[i - 1]}\n\n")

            _ = f.write("=== Conversation ===\n\n")

            for i, msg in enumerate(self._conversation_log):
                if i > 0:
                    _ = f.write("\n" + "\u2500" * 80 + "\n\n")

                _ = f.write(f"{msg['agent']}: {msg['content']}\n")

    def run_conversation(self) -> Iterator[tuple[str, Iterator[str]]]:
        """Generate an iterator of conversation responses.

        Yields:
            (str, Iterator[str]): A tuple of the agent name and an iterator of the accumulated response.

                                  Note: The iterator returns the entire message up until the newest chunk received,
                                  not just the new chunk.

                                  For example, if the first iteration yields "Hello, ", the second iteration will yield
                                  "Hello, world!" instead of just "world!".
        """

        def add_agent_response(agent_idx: int, response: dict[str, Any]) -> None:
            """Add a message from an agent to the conversation log and the agents' message history.

            Args:
                agent_idx (int): Index of the agent the message is from
                message (str): Message content
            """
            message: str = response["message"]
            # Message with dialogue marker to indicate which agent is speaking.
            message_with_marker = f"{self.agents[agent_idx].name}: {message}"

            # The agents should get the full JSON response as context, to reinforce the response format and help them
            # generate more coherent responses.
            for i, agent in enumerate(self.agents):
                agent.add_message(
                    self.agents[agent_idx].name,
                    # Use "assistant" instead of "user" for the agent's own messages.
                    "assistant" if i == agent_idx else "user",
                    # For the agent's own messages, use the full JSON response to reinforce the response format.
                    # For other agents' messages, use the message content with the dialogue marker.
                    str(response) if i == agent_idx else message_with_marker,
                )

            # If a moderator agent is present, add the message to the moderator's message history.
            if self.moderator is not None:
                self.moderator.add_message(self.agents[agent_idx].name, "user", message_with_marker)

            # For the conversation log, only the message content is needed.
            self._conversation_log.append({"agent": self.agents[agent_idx].name, "content": message})

        agent_idx: int = self._pick_next_agent(None)

        # If a non-empty initial message is provided, start with it.
        if self.initial_message is not None:
            # Make the first agent the one to say the initial message, and the second agent the one to respond.
            add_agent_response(agent_idx, {"message": self.initial_message})
            yield (self.agents[agent_idx].name, iter([self.initial_message]))
            agent_idx = self._pick_next_agent(agent_idx)

        while True:
            current_agent = self.agents[agent_idx]
            response_stream = current_agent.get_response(self._output_format)

            # Will be populated with the full JSON response once the response stream is exhausted.
            response_json: dict[str, Any] = {}

            def parse_partial_json(json_string: str) -> dict[str, Any]:
                """Parse a partial JSON response using the partial JSON parser, and return the JSON object."""
                # Don't use `partial_json_parser.loads()` directly because it doesn't have good type hints.
                return cast(dict[str, Any], json.loads(ensure_json(json_string)))

            def stream_chunks() -> Iterator[str]:
                nonlocal response_json

                response: str = ""

                # Accumulate chunks until the message field is found in the JSON response.
                for response_chunk in response_stream:
                    response += response_chunk
                    response_json = parse_partial_json(response)

                    if "message" in response_json:
                        break

                # Message field is found, yield the entire message gradually as new chunks arrive.
                for response_chunk in response_stream:
                    response += response_chunk
                    response_json = parse_partial_json(response)

                    yield response_json["message"]

            yield (current_agent.name, stream_chunks())

            add_agent_response(agent_idx, response_json)

            # Check if the conversation should be terminated.
            if response_json.get("terminate", False):
                break

            agent_idx = self._pick_next_agent(agent_idx)

    def _create_moderator_agent(self) -> None:
        moderator_agent_model: str | None = None
        moderator_agent_ctx_size: int | None = None
        lowest_param_count: int | None = None

        # Find the model with the lowest parameter count to use as the moderator agent.
        # Also use the highest context size among the agents.
        for agent in self.agents:
            model_param_count: int = agent.get_param_count()

            if lowest_param_count is None or model_param_count < lowest_param_count:
                moderator_agent_model = agent.model
                lowest_param_count = model_param_count

            if moderator_agent_ctx_size is None or agent.ctx_size > moderator_agent_ctx_size:
                moderator_agent_ctx_size = agent.ctx_size

        assert moderator_agent_model is not None and moderator_agent_ctx_size is not None

        self.moderator = AIAgent(
            name="Moderator",
            model=moderator_agent_model,
            temperature=0.8,
            ctx_size=moderator_agent_ctx_size,
            system_prompt="You are the conversation moderator. Your task is to analyze the conversation "
            + "and choose who speaks next. You should prioritize giving each character an equal opportunity to speak. "
            + "Most importantly, you should prioritize keeping the conversation entertaining and engaging.",
        )

    def _pick_next_agent(self, current_agent_idx: int | None) -> int:
        """Pick the next agent to speak based on the turn order.

        The different turn order strategies are as follows:
        - "round_robin": Cycle through the agents in order.
        - "random": Randomly pick an agent to speak next.
        - "chain": The agent that just spoke picks the next agent to speak.
        - "moderator": A moderator agent picks the next agent to speak.
        - "vote": Each agent votes for the next agent to speak, and the agent with the most votes is chosen.
                  In case of a tie, one of the tied agents is chosen randomly.

        Args:
            current_agent_idx (int | None): Index of the agent that just spoke. Should be None if no agent has spoken
                                            yet.

        Returns:
            int: Index of the agent to speak next
        """
        # Only two agents, so the next agent is the other one.
        if len(self.agents) == 2 and current_agent_idx is not None:
            return 1 if current_agent_idx == 0 else 0

        def choice_enum(ignore_idx: list[int]) -> enum.Enum:
            choices_dict: dict[str, str] = {}

            choices_dict = {agent.name: agent.name for i, agent in enumerate(self.agents) if i not in ignore_idx}

            return enum.Enum("NextAgentChoices", choices_dict)  # pyright: ignore[reportReturnType]

        match self.turn_order:
            case "round_robin":
                return (current_agent_idx + 1) % len(self.agents) if current_agent_idx is not None else 0
            case "random":
                if current_agent_idx is None:
                    return random.randint(0, len(self.agents) - 1)

                idx = random.randint(0, len(self.agents) - 2)
                return idx if idx < current_agent_idx else idx + 1
            case "chain":
                if current_agent_idx is None:
                    # No agent has spoken yet, so pick a random agent to start the conversation.
                    return random.randint(0, len(self.agents) - 1)

                agent_choices_enum = choice_enum([current_agent_idx])

                chain_output_format: type[BaseModel] = create_model(
                    "ChainOutputFormat",
                    next_agent=(agent_choices_enum, Field(description="Name of the next character to speak")),
                )

                response = "".join(list(self.agents[current_agent_idx].get_response(chain_output_format)))
                next_agent_name = json.loads(response)["next_agent"]

                return self._agent_name_to_idx[next_agent_name]
            case "moderator":
                assert self.moderator is not None
                agent_choices_enum = choice_enum([current_agent_idx] if current_agent_idx is not None else [])

                moderator_output_format: type[BaseModel] = create_model(
                    "ModeratorOutputFormat",
                    next_agent=(agent_choices_enum, Field(description="Name of the next character to speak")),
                )

                moderator_response = "".join(list(self.moderator.get_response(moderator_output_format)))
                next_agent_name = json.loads(moderator_response)["next_agent"]

                return self._agent_name_to_idx[next_agent_name]
            case "vote":
                agent_votes: dict[str, int] = {agent.name: 0 for agent in self.agents}

                for i, agent in enumerate(self.agents):
                    agent_choices_enum = choice_enum([i] if current_agent_idx is None else [current_agent_idx, i])

                    vote_output_format: type[BaseModel] = create_model(
                        "VoteOutputFormat",
                        next_agent=(agent_choices_enum, Field(description="Name of the next character to speak")),
                    )

                    response = "".join(list(agent.get_response(vote_output_format)))
                    agent_name = json.loads(response)["next_agent"]

                    assert agent_name in agent_votes, f"Invalid agent name: {agent_name}"
                    agent_votes[agent_name] += 1

                # Find the agents with the most votes and pick one of them randomly.
                max_votes = max(agent_votes.values())
                agents_with_max_votes = [agent_name for agent_name, votes in agent_votes.items() if votes == max_votes]

                return self._agent_name_to_idx[random.choice(agents_with_max_votes)]
