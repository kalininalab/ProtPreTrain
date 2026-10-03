# RINDTI contribution guide

The `pyproject.toml` file defines the formatting guidelines for the project.

Please use [pre-commit](https://pre-commit.com/) to run the linter and the tests before committing.

Set up the environment with `uv sync`, then run `pre-commit install` once to enable the hooks (ruff, ruff-format, interrogate). Run the smoke tests with `python -m pytest`.

After that, on every commit the code will be formatted and some basic checks will be run.

The commit will be rejected if the documentation coverage is below 80% (this is important for the automatic documentation building).
