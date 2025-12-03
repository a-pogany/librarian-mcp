# Repository Guidelines

## Project Structure & Module Organization
- `backend/main.py` launches the MCP HTTP/SSE server; most logic lives under `backend/core` (`indexer.py`, `parsers.py`, `search.py`) and MCP bindings in `backend/mcp_server`.
- Tests sit in `backend/tests` mirroring core modules (`test_indexer.py`, `test_parsers.py`, `test_search.py`) with shared fixtures in `conftest.py`.
- Configuration defaults are in the repo root `config.json`; environment handling and logging live in `backend/config/settings.py`.
- Documentation content to index belongs under `docs/` following the product/component hierarchy shown in README.

## Build, Test, and Development Commands
- Create env: `python3 -m venv venv && source venv/bin/activate`.
- Install deps: `pip install -r backend/requirements.txt`.
- Run server: `cd backend && python main.py` (serves on `127.0.0.1:3001/mcp` by default).
- Run tests: `cd backend && pytest` or `pytest --cov=core --cov=mcp_server --cov-report=html` for coverage output in `htmlcov/`.
- Lint/format: follow PEP 8; if adding tools, prefer `ruff`/`black` and document the command.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints on public interfaces, module and file names in `snake_case`, classes in `CapWords`, tests in `test_<unit>.py::test_<behavior>`.
- Keep functions small and pure where possible; isolate I/O in thin adapters (e.g., MCP handlers or file watchers).
- Log through the configured logger from `settings.setup_logging`; avoid print statements in runtime code.

## Testing Guidelines
- Favor unit tests in `backend/tests` that cover parsing, indexing, and search edge cases (encoding errors, large files, empty results).
- Use `pytest.mark.slow` or `pytest.mark.integration` for heavier runs; keep default suite fast.
- When adding fixtures that touch the filesystem, write to temporary directories and clean up via pytest fixtures.

## Commit & Pull Request Guidelines
- Commit messages: short imperative subjects (e.g., `Add docx parser fallback`), include a brief body when rationale is non-obvious.
- Pull requests should summarize behavior changes, list test commands run, and reference related issues/tasks. Add screenshots or sample queries if changing MCP responses.
- Keep diffs focused; prefer separate PRs for refactors vs. features. Update docs (`README.md`, `QUICKSTART.md`, or `AGENTS.md`) when workflows change.

## Configuration & Operations Notes
- Runtime config reads from `config.json` with env overrides: `DOCS_ROOT_PATH`, `MCP_HOST`, `MCP_PORT`, `LOG_LEVEL`.
- Ensure the watched docs directory exists before starting the server; avoid committing large or proprietary documents to `docs/`.
- Log output defaults to `mcp_server.log`; rotate or redirect if running long-lived processes.***
