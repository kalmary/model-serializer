from collections.abc import Callable
from pathlib import Path
from typing import Any


def validate_serializer(
    mode: str,
    observed_metric: Callable[..., list[float]],
    n_models: int,
    model_name: str,
    configs: dict[str, Any],
) -> None:
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    if not callable(observed_metric):
        raise TypeError("observed_metric must be callable")
    if isinstance(n_models, bool) or not isinstance(n_models, int):
        raise TypeError("n_models must be an integer")
    if n_models < 1:
        raise ValueError("n_models must be positive")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if (
        not model_name.strip()
        or model_name in (".", "..")
        or "/" in model_name
        or "\\" in model_name
    ):
        raise ValueError("model_name must be a single non-empty path component")
    if not isinstance(configs, dict):
        raise TypeError("configs must be a dictionary")


def validate_output_path(path: Path) -> None:
    current_path = Path.cwd().resolve()
    if path == current_path or path in current_path.parents:
        raise ValueError("checkpoint path cannot contain the current directory")
