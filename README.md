# LLM Conversation Tool

A Python application that enables conversations between multiple LLM agents using various providers (OpenAI, Ollama, Anthropic, and more). The agents can engage in back-and-forth dialogue with configurable parameters and models.

## Features

- Support for multiple LLM providers:
  - Ollama (local models)
  - OpenAI (GPT-5, GPT-5-mini, GPT-5-nano, o4-high, etc.)
  - Anthropic (Claude)
  - Google (Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash, etc.)
  - OpenRouter, Together, Groq, DeepSeek, and any other provider with an OpenAI compatible API.
- Flexible configuration via JSON file or interactive setup
- Multiple conversation turn orders (round-robin, random, chain, moderator, vote)
- Conversation logging and export (text or JSON format)
- Agent-controlled conversation termination (needs to be enabled)
- Markdown formatting support (needs to be enabled)

## Installation

### Prerequisites

- Python 3.13
- Ollama for local models, or API credentials for your chosen LLM provider

### How to Install

#### From PyPI

The project is available on PyPI. You can install it using:

```bash
pip install llm-conversation
```

#### From Source

If you prefer to install from the source code, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/famiu/llm_conversation.git
    cd llm_conversation
    ```

2.  **Create and activate a virtual environment:**
    It is highly recommended to use a virtual environment to manage dependencies.

    ```bash
    uv venv
    source .venv/bin/activate  # On macOS/Linux
    .\.venv\Scripts\activate   # On Windows
    ```

3.  **Install the project in editable mode:**
    This will install all the required dependencies and link the project directly to your virtual environment.

    ```bash
    uv pip install -e .
    ```

After these steps, the `llm_conversation` package and its dependencies will be installed and ready to use within your active virtual environment.

## Usage

### Command Line Arguments

```txt
llm-conversation [-h] [-V] [-o OUTPUT] [-c CONFIG]

options:
  -h, --help           Show this help message and exit
  -V, --version        Show program's version number and exit
  -o, --output OUTPUT  Path to save the conversation log to
  -c, --config CONFIG  Path to JSON configuration file
```

### Interactive Setup

If no configuration file is provided, the program will guide you through an intuitive interactive setup process.

### Configuration File

You can provide a JSON configuration file using the `-c` flag for reproducible conversation setups.

#### Example Configuration

```json
{
  "providers": {
    "openai": {
      "api_key": "your-api-key-here"
    },
    "anthropic": {
      "api_key": "your-api-key-here"
    }
  },
  "agents": [
    {
      "name": "Claude",
      "provider": "anthropic",
      "model": "claude-sonnet-4-20250514",
      "temperature": 0.9,
      "ctx_size": 4096,
      "system_prompt": "You are extremely paranoid about everything and constantly question others' intentions."
    },
    {
      "name": "G. Pete",
      "provider": "openai",
      "model": "gpt-5",
      "temperature": 1,
      "ctx_size": 4096,
      "system_prompt": "You are the laziest person ever. You respond as briefly as possible, and constantly complain about having to work."
    },
    {
      "name": "Liam",
      "provider": "ollama",
      "model": "llama3.2",
      "temperature": 0.7,
      "ctx_size": 2048,
      "system_prompt": "You are easily irritable and quick to anger."
    }
  ],
  "settings": {
    "initial_message": "THEY are out to get us",
    "use_markdown": true,
    "allow_termination": true,
    "turn_order": "round_robin"
  }
}
```

#### Provider Configuration

The `providers` section defines API endpoints and credentials:

- **base_url**: The API endpoint URL (optional for built-in providers)
- **api_key**: Authentication key (can be omitted for local providers like Ollama)

Built-in providers (base_url automatically configured):

- `ollama`: Local Ollama models
- `openai`: OpenAI GPT models
- `anthropic`: Anthropic Claude models
- `google`: Google Gemini models
- `openrouter`: OpenRouter proxy service
- `together`: Together AI models
- `groq`: Groq inference service
- `deepseek`: DeepSeek models

For built-in providers, you only need to specify the `api_key`. Custom providers require both `base_url` and `api_key`.

#### Agent Configuration

Each agent in the `agents` array requires:

- **name**: Unique identifier for the agent
- **provider**: Reference to a provider defined in the `providers` section
- **model**: The specific model to use (e.g., "gpt-4", "llama3.2", "claude-3-sonnet")
- **system_prompt**: Instructions defining the agent's behavior and personality

Optional parameters:

- **temperature** (0.0-1.0, default: 0.8): Controls response creativity/randomness
- **ctx_size** (default: 2048): Maximum context window size

#### Conversation Settings

The `settings` section controls conversation behavior:

- **initial_message** (optional): Starting message for the conversation
- **use_markdown** (default: false): Enable Markdown formatting in responses
- **allow_termination** (default: false): Allow agents to end conversations
- **turn_order** (default: "round_robin"): Agent selection strategy:
  - `"round_robin"`: Cycle through agents in order
  - `"random"`: Randomly select next agent
  - `"chain"`: Current agent chooses next speaker
  - `"moderator"`: Dedicated moderator selects speakers
  - `"vote"`: All agents vote for next speaker
- **moderator** (optional): Custom moderator agent configuration

You can take a look at the [JSON configuration schema](schema.json) for more details.

### Running the Program

1. **Interactive setup** (prompts for configuration):

   ```bash
   llm-conversation
   ```

2. **Using a configuration file**:

   ```bash
   llm-conversation -c config.json
   ```

3. **Saving conversation to a file**:
   ```bash
   llm-conversation -c config.json -o conversation.txt
   ```
4. **JSON output format**:
   ```bash
   llm-conversation -c config.json -o conversation.json
   ```

### Conversation Controls

The conversation will continue until:

- An agent terminates the conversation (if termination is enabled)
- The user interrupts with `Ctrl+C`

## Output Format

When saving conversations, the output file includes:

- Configuration details for both agents
- Complete conversation history with agent names and messages

Additionally, if the output file has a `.json` extension, the output will automatically have JSON format.

## Contributing

Feel free to submit issues and pull requests for bug fixes or new features. Do keep in mind that this is a hobby project, so please have some patience.

## License

This software is licensed under the GNU Affero General Public License v3.0 or any later version. See [LICENSE](LICENSE) for more details.
