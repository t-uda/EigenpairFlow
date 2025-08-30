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
