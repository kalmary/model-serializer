from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch

from model_serializer.artifacts import load_checkpoint_model
from model_serializer.local import ModelSerializer
from model_serializer.metrics import Metrics
from model_serializer.ranking import checkpoint_id
from model_serializer.tracking.mlflow_client import (
    create_client,
    get_or_create_experiment,
    log_checkpoint,
)


class MLflowModelSerializer[MetricsType: Metrics]:
    def __init__(
        self,
        mode: Literal["min", "max"],
        observed_metric: Callable[[MetricsType], list[float]],
        n_models: int,
        model_name: str,
        configs: dict[str, Any],
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="model_serializer_mlflow_"
        )
        self._local = ModelSerializer[MetricsType](
            mode=mode,
            observed_metric=observed_metric,
            n_models=n_models,
            path=Path(self._temporary_directory.name) / "checkpoints",
            model_name=model_name,
            configs=configs,
        )
        self._client = create_client(tracking_uri)
        self._experiment_id = get_or_create_experiment(
            self._client, experiment_name, artifact_location
        )
        self._runs: dict[int, str] = {}
        self.model_name = model_name
        self.configs = self._local.configs

    def checkpoint(self, model: Any, metrics: MetricsType) -> str | None:
        checkpoint_path = self._local.checkpoint(model, metrics)
        if checkpoint_path is None:
            return None

        unique_id = checkpoint_id(checkpoint_path)
        run_id = log_checkpoint(
            self._client,
            self._experiment_id,
            checkpoint_path,
            self.model_name,
            unique_id,
            self._local.observed_metric(metrics)[-1],
            self.configs,
        )
        self._runs[unique_id] = run_id
        self._synchronize_ranking()
        return run_id

    def _synchronize_ranking(self) -> None:
        ranked_paths = sorted(self._local.path.iterdir())
        retained_ids = {checkpoint_id(path) for path in ranked_paths}
        for removed_id in self._runs.keys() - retained_ids:
            self._client.delete_run(self._runs.pop(removed_id))
        for rank, path in enumerate(ranked_paths, start=1):
            self._client.set_tag(
                self._runs[checkpoint_id(path)], "serializer.rank", str(rank)
            )

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> MLflowModelSerializer[MetricsType]:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def load_mlflow_model(
    run_id: str,
    model_factory: Callable[[dict[str, Any]], torch.nn.Module],
    tracking_uri: str | None = None,
    destination: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    client = create_client(tracking_uri)
    downloaded = client.download_artifacts(
        run_id,
        "",
        dst_path=str(Path(destination).expanduser().resolve())
        if destination is not None
        else None,
    )
    return load_checkpoint_model(
        downloaded, model_factory, map_location=map_location
    )
