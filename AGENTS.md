## Style
- Classes in PascalCase; functions, variables, and modules in snake_case; constants in UPPER_SNAKE_CASE.
- Descriptive and full names. No vague names like `data`, `tmp`, `res`, or `x2`.
- **No magic numbers**: use a named constant within the module that uses it.
- **No unnecessary defensive programming**: no "just in case" `try/except` blocks, and no input validation already guaranteed by the caller or type hints.
- **No speculative abstractions**: no base classes, no factories, and no single-implementation interfaces.
- Prefer simple functions over classes if there is no state.
- **Type hints** required on all signatures.
- **Public methods must have docstrings in numpydoc format**. Optionally, also document private methods if they are complex.

## Commands
- **Environment:** `uv sync`
- **Lint/Format:** `uv run ruff check --fix . && uv run ruff format .`

## Prohibited
- Editing `uv.lock` manually.
- Disabling ruff rules with `# noqa`.