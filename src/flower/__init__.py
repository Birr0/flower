from __future__ import annotations

from importlib.metadata import version

from . import (
    data,
    evaluation,
    explainability,
    inference,
    models,
    outliers,
    training,
)

__all__ = (
    "__version__",
    "data",
    "evaluation",
    "explainability",
    "inference",
    "models",
    "outliers",
    "training",
)
__version__ = version(__name__)
