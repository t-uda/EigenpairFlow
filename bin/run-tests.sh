#!/usr/bin/env bash
# This script ensures that pytest is run within the project's poetry environment.

# Ensure the script exits if any command fails
set -e

# Check if poetry is available on the PATH, if not, try a common location.
if ! command -v poetry &> /dev/null; then
    # Add the common poetry install location to the PATH
    export PATH="$HOME/.local/bin:$PATH"
    # Check again. If still not found, exit with an error.
    if ! command -v poetry &> /dev/null; then
        echo "Error: 'poetry' command not found." >&2
        echo "Please ensure Poetry is installed and its location is in your PATH." >&2
        exit 1
    fi
fi

# Execute pytest within the poetry environment
poetry run pytest
