import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml


def validate_model(model: Any) -> None:
    if not callable(getattr(model, "state_dict", None)):
        raise TypeError("model must provide a callable state_dict method")


def write_checkpoint_artifacts(
    path: Path,
    model: Any,
    metric_values: list[tuple[str, list[float]]],
    configs: dict[str, Any],
) -> None:
    torch.save(model.state_dict(), path / "model.pt")
    names = [name for name, _ in metric_values]
    histories = [history for _, history in metric_values]
    with (path / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(names)
        writer.writerows(zip(*histories, strict=True))
    with (path / "configs.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(configs, config_file, sort_keys=False, allow_unicode=True)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    model_factory: Callable[[dict[str, Any]], torch.nn.Module],
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    path = Path(checkpoint_path).expanduser().resolve()
    with (path / "configs.yaml").open(encoding="utf-8") as config_file:
        configs = yaml.safe_load(config_file)
    if not isinstance(configs, dict):
        raise ValueError("checkpoint configs must be a mapping")
    model = model_factory(configs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_factory must return a torch module")
    state_dict = torch.load(
        path / "model.pt", map_location=map_location, weights_only=True
    )
    model.load_state_dict(state_dict)
    return model
