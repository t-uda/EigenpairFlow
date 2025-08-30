This file provides instructions for AI agents working on this repository.

## Development Workflow

Before committing any changes, please follow these steps to ensure code quality and consistency.

1.  **Install dependencies:**
    If this is your first time working on the project, or if dependencies have changed, install them using Poetry:
    ```bash
    poetry install
    ```

2.  **Format your code:**
    Automatically format your code using `ruff`.
    ```bash
    poetry run ruff format .
    ```

3.  **Lint your code:**
    Check for and automatically fix any linting issues with `ruff`.
    ```bash
    poetry run ruff check . --fix
    ```

4.  **Run tests:**
    Execute the test suite using `pytest` to ensure your changes haven't introduced any regressions.
    ```bash
    poetry run pytest
    ```

5.  **Commit your changes:**
    Once all checks pass, you can commit your changes.

## Language and Style

*   **Docstrings:** Docstrings must be written in **Japanese**. They can include simple English terms where appropriate.
*   **Comments:** Code comments other than docstrings can be written in English.
