name: CI
on: push to [main, test/add-tests] + PRs to [main]

Two jobs, both ubuntu-latest, run in parallel:

1. lint
| Step                                   | What it does                                                    |
|----------------------------------------|-----------------------------------------------------------------|
| actions/checkout@v4                    | Checks out the repo                                             |
| actions/setup-python@v5                | Installs Python 3.12                                            |
| astral-sh/setup-uv@v5                  | Installs uv (fast pip replacement)                              |
| uv sync --group dev                    | Installs dev deps from pyproject.toml (ruff, pytest, etc.)      |
| uv run ruff check src/ tests/          | Static analysis: unused imports, bad patterns, long lines, etc. |
| uv run ruff format --check src/ tests/ | Verifies code is formatted according to ruff defaults           |

2. test
| Step                                                      | What it does                                           |
|-----------------------------------------------------------|--------------------------------------------------------|
| actions/checkout@v4                                       | Checks out the repo                                    |
| actions/setup-python@v5                                   | Installs Python 3.12                                   |
| astral-sh/setup-uv@v5                                     | Installs uv                                            |
| uv sync --group dev                                       | Installs dev deps                                      |
| uv run pytest tests/ -v --cov=src/flower --cov-report=xml | Runs all 134 tests verbose + generates coverage report |
| codecov/codecov-action@v3                                 | Uploads coverage XML to Codecov (only on push to main) |

Notable design choices:
- No needs: lint on the test job — test results always show up even when lint fails (the source code currently has ~100 pre-existing lint errors).
- uv run prefix — runs tools inside the venv without needing source .venv/bin/activate.
- Codecov upload gated to main pushes only, so PRs don't spam the dashboard.
