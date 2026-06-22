from pathlib import Path


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
