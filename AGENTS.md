This file provides instructions for AI agents working on this repository.

## Development Workflow with pre-commit

This repository uses `pre-commit` to automatically run code formatters and linters before each commit. This ensures code quality and consistency.

### 1. One-Time Setup

Before you start working, you need to install the dependencies and the pre-commit hooks. Run these commands from the root of the repository:

```bash
# Install project dependencies, including development tools
poetry install

# Install the pre-commit hooks
pre-commit install
```

### 2. Development Workflow

1.  **Edit Code:** Make your changes to the source code as required by your task.

2.  **Commit Changes:** Once you are ready, commit your changes using `git commit`. The pre-commit hooks will run automatically.

    *   **If hooks pass:** Your commit will be created successfully.
    *   **If hooks fail:** The commit will be aborted. You must fix the issues reported by the hooks before you can commit.

3.  **Handling Hook Failures:**
    *   **Formatting Failures (`ruff-format`):** The formatter will automatically change your files. These changes are not yet staged. You need to stage them with `git add <changed_files>` and then run `git commit` again.
    *   **Linting Failures (`ruff`):** The linter will report issues in your code. Read the error messages and fix the code accordingly. After fixing, stage your changes and commit again.
    *   **Test Failures (`pytest`):** The test suite will run automatically. If any tests fail, the commit will be blocked. Read the `pytest` output to understand which tests failed, debug the code, and fix the issues until all tests pass. Then, commit again.

4.  **Verify Changes**:
    Before finalizing your work, review your changes to ensure they are within the scope of the assigned task and do not include any unintended modifications. A good way to do this is to review the output of `git diff`.

## Language and Style

*   **Coding Style:** All Python code should adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/). The `ruff` pre-commit hook helps enforce this.
*   **Docstrings:** Docstrings must be written in **Japanese**. They can include simple English terms where appropriate.
*   **Comments:** Code comments other than docstrings can be written in English.
