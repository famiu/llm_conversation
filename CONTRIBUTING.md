# Contributing to LLM Conversation Tool

Thank you for your interest in contributing! Whether you're a developer or just a user who wants to help improve the project, there are many ways for you to contribute.

## For Non-Developers: Reporting Issues

You don't need to be a programmer to help improve this project. Reporting bugs, suggesting features, and providing feedback are invaluable contributions.

### How to Report a Bug

When reporting a bug, please include:

1. **Clear title**: Briefly describe the problem
2. **Steps to reproduce**: Exact steps that cause the issue
3. **Expected behavior**: What you thought should happen
4. **Actual behavior**: What actually happened
5. **Environment details**:
   - Operating system (Windows, macOS, Linux)
   - Python version (`python --version`)
   - Package version (`llm-conversation --version`)
   - Ollama version if relevant
6. **Configuration**: Include your conversation config file if relevant (remove any sensitive information)
7. **Error messages**: Copy the full error message if any
8. **Screenshots**: If relevant, especially for UI issues

### Feature Requests

When suggesting a feature:

1. **Describe the feature**: What should it do?
2. **Use case**: Why would this be useful?
3. **Examples**: How would you use it?
4. **Alternatives**: Have you considered other solutions?

### Questions and Support

- Check existing issues first to avoid duplicates
- Use clear, descriptive titles
- Be patient - this is a hobby project maintained in free time
- Be respectful and constructive in your communication

## For Developers: Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Ollama installed and running for testing

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/famiu/llm_conversation.git
   cd llm_conversation
   ```

2. Install development dependencies:
   ```bash
   uv sync
   ```

## Development Commands

- **Install dependencies**: `uv sync`
- **Format code**: `uv run ruff format` (use `--check` to only check for changes)
- **Lint code**: `uv run ruff check` (use `--fix` to auto-fix)
- **Type check**: `uv run ty check`
- **Run all checks**: `uv run ruff format --check && uv run ruff check && uv run ty check`

## Code Style Guidelines

### Python Version
- **Python 3.13+** only

### Formatting and Style
- **Line length**: 120 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Formatting**: Use Ruff formatter with Google docstring convention
- **Import order**: Standard library, third-party, local imports (prefer `from` imports for clarity)

### Type Annotations
- Always use type hints, prefer explicit types over `Any`
- Use modern Python 3.9+ built-in types (`list`, `dict`, `set`, `tuple`) instead of `typing` module equivalents (`List`, `Dict`, `Set`, `Tuple`)

### Naming Conventions
- **Functions and variables**: snake_case
- **Classes**: PascalCase
- **Constants**: UPPER_CASE

### Documentation and Comments
- Use Google-style docstrings for functions and classes
- Avoid inline comments unless critical for understanding
- Write clear, concise docstrings that explain the "why" not just the "what"
- **Always update documentation** when adding new features, changing behavior, or modifying APIs

### Error Handling
- Use specific exception types rather than generic `Exception`
- Fail fast, prefer early returns over deeply nested conditionals

## Debugging and Logging

The application includes developer logging that can be enabled via environment variables:

- **`LLM_CONVERSATION_LOG_LEVEL`**: Set to `DEBUG`, `INFO`, `WARNING` or `ERROR` to enable logging (disabled by default)
- **`LLM_CONVERSATION_LOG_FILE`**: Optional file path for log output (defaults to stderr only)

### Example usage:
```bash
# Enable debug logging to stderr
LLM_CONVERSATION_LOG_LEVEL=DEBUG llm-conversation

# Enable info logging to a file
LLM_CONVERSATION_LOG_LEVEL=INFO LLM_CONVERSATION_LOG_FILE=debug.log llm-conversation
```
