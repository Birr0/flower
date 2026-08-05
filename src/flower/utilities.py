import os
from pathlib import Path

from dotenv import load_dotenv


def data_root() -> str:
    """Resolve ``DATA_ROOT`` at call time.

    Deliberately a function rather than a module constant. Reading the variable
    at module scope makes importing a data module fail on any machine without a
    populated ``.env`` -- including a fresh clone and CI -- before any user code
    runs, and the failure surfaces far from its cause.

    ``.env`` is loaded here rather than at import for the same reason: importing
    ``flower.data.*`` should have no side effects.

    Returns:
        str: ``DATA_ROOT`` with any trailing separator stripped, so callers can
        join onto it without producing a doubled separator.

    Raises:
        RuntimeError: If ``DATA_ROOT`` is unset, naming the variable rather than
            letting ``None`` propagate into a path.
    """
    load_dotenv()
    root = os.getenv("DATA_ROOT")
    if not root:
        msg = (
            "DATA_ROOT is not set. Add it to .env at the project root or export "
            "it before loading a dataset -- see the Setup section of README.md."
        )
        raise RuntimeError(msg)
    return root.rstrip("/")


def find_project_root(start_path: str = __file__) -> str:
    """
    Find the root directory of the project by looking for a 'pyproject.toml' file.

    Args:
        start_path (str): The starting path to begin the search. Defaults to the
                          path of the current file.

    Returns:
        str: The path to the project root directory.

    Raises:
        FileNotFoundError: If the project root is not found.
    """
    current_path = Path(start_path).resolve()
    while current_path != current_path.parent:  # Traverse up to the root
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
    msg = "Project root with 'pyproject.toml' not found."
    raise FileNotFoundError(msg)
