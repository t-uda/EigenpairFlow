This file provides instructions for AI agents working on this repository.

## Development Workflow with pre-commit

This repository uses `pre-commit` to automatically run code formatters and linters before each commit. This ensures code quality and consistency.

### 1. One-Time Setup

Before you start working, set up your environment by running the following commands from the repository root. This ensures that dependencies are installed and `pre-commit` hooks are correctly configured for automated code quality checks.

```bash
# 1. Install project dependencies
poetry install

# 2. Prevent a common git hook error before installation
git config --global --unset core.hooksPath

# 3. Install the pre-commit hooks into your local .git/hooks/ directory
poetry run pre-commit install
```

### Important Notes for Agents

- **Environment Persistence**: The setup commands above must be run in any new terminal session or environment. The pre-commit hooks will not work if they are not installed in the current context.
- **Verify Installation**: After running the setup, you **must** verify that the hooks are working correctly by running the following command:
  ```bash
  pre-commit run --all-files
  ```
  This command will run all configured checks against all files in the repository. It should pass without errors on a clean checkout. If it fails, it indicates a problem with the setup or the existing codebase that must be resolved before proceeding.

### A Note on Managing Dependencies

If you need to add, remove, or update dependencies in `pyproject.toml`, you must also update the `poetry.lock` file to reflect these changes. After modifying `pyproject.toml`, run the following command:

```bash
poetry lock
```

This ensures that the project's dependencies remain consistent and reproducible. After running the command, remember to commit both the `pyproject.toml` and `poetry.lock` files.

### 2. Development Workflow

1.  **Edit Code:** Make your changes to the source code as required by your task.

2.  **Commit Changes:** Once you are ready, commit your changes using `git commit`. The pre-commit hooks will run automatically.

    *   **If hooks pass:** Your commit will be created successfully.
    *   **If hooks fail:** The commit will be aborted. You must fix the issues reported by the hooks before you can commit.

3.  **Handling Hook Failures:**
    *   **Formatting Failures (`ruff-format`):** The formatter will automatically change your files. These changes are not yet staged. You need to stage them with `git add <changed_files>` and then run `git commit` again.
    *   **Linting Failures (`ruff`):** The linter will report issues in your code. Read the error messages and fix the code accordingly. After fixing, stage your changes and commit again.
    *   **Test Failures (`pytest`):** The test suite will run automatically. If any tests fail, the commit will be blocked. Read the `pytest` output to understand which tests failed, debug the code, and fix the issues until all tests pass. Then, commit again.
    *   **Special Case: Hook Deadlocks:** In some situations, a hook might fail not because of a code issue, but because of an environment problem (e.g., `poetry` not being found in the `PATH`). This can block all file modifications. If you suspect this is happening, you can temporarily skip a specific hook by setting the `SKIP` environment variable before invoking a tool. For the `pytest` hook, you would set it like this: `SKIP=pytest`. This should be used as a last resort to resolve a stuck state.

4.  **Verify Changes**:
    Before finalizing your work, review your changes to ensure they are within the scope of the assigned task and do not include any unintended modifications. A good way to do this is to review the output of `git diff`.

## Notebook Development

When working with notebooks for verification or demonstration, follow these guidelines:

-   Add any necessary notebook development dependencies to the `[tool.poetry.group.dev.dependencies]` section in `pyproject.toml`.
-   Ensure the development environment can be set up by running `poetry install`.
-   The CI/CD pipeline does not run or test notebooks, but including tools for a better local development experience is encouraged.
-   To make the Poetry environment's kernel available in Jupyter, add `ipykernel` as a dev dependency. You can then register the kernel using a command like `poetry run python -m ipykernel install --user --name=eigenpairflow`.

### How to Use the Notebook Environment

1.  **Launch JupyterLab:**
    ```bash
    poetry run jupyter lab
    ```

2.  **Select the Kernel:**
    When creating a new notebook in JupyterLab, be sure to select the `eigenpairflow` kernel. This ensures that your notebook runs in the project's Poetry environment and has access to all required dependencies.
### Using Papermill for Automated Execution
`papermill` is a tool for running notebooks programmatically. It is particularly useful for automation, reproducibility, and parameterizing notebooks.
-   **Parameterization:** To make a notebook parameterizable, create a cell with the tag `parameters`. Variables in this cell can be overridden from the command line.
-   **Execution:** You can run a notebook and save the output version (with all cell outputs) using the following command structure.
**Example:**
```bash
poetry run papermill \\
  notebooks/your_notebook.ipynb \\
  notebooks/output.ipynb \\
  -p alpha 0.5 \\
  -p model_type "fancy_model"
```
This command executes `your_notebook.ipynb`, sets the `alpha` parameter to `0.5` and `model_type` to `"fancy_model"`, and saves the resulting notebook to `output.ipynb`.

## Language and Style

*   **Coding Style:** All Python code should adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/). The `ruff` pre-commit hook helps enforce this.
*   **Docstrings:** Docstrings must be written in **Japanese**. They can include simple English terms where appropriate.
*   **Comments:** Code comments other than docstrings can be written in English.
