# Agent Development Guidelines

## Development Setup and Guidelines

For complete development setup, code style guidelines, and contribution process, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Key Points for Agents

- **Python version**: 3.13+ only
- **Project structure**: Source in `src/llm_conversation/`, scripts in `scripts/`
- **Documentation**: Always update README.md and docstrings for functional changes. For changes concerning development, update CONTRIBUTING.md instead.
- **Checks**: After major changes, format the code and run all checks with `uv run ruff format && uv run ruff check && uv run ty check`
