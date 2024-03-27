#!/bin/bash


PYTHON_FILES=$(git ls-files '*.py')

if [ -z "$PYTHON_FILES" ]; then
    echo "No Python files found to check."
    exit 0
fi

run_tool() {
    echo ">> $1..."
    $1 $2 $PYTHON_FILES
}

# Static type checker
run_tool mypy "--strict"

# Sort imports
run_tool isort "check"

# Fast, extensible Python linter
run_tool ruff "check"


echo "All checks and formatting done."
