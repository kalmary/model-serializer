import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


def create_client(tracking_uri: str | None) -> Any:
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def get_or_create_experiment(
    client: Any,
    experiment_name: str,
    artifact_location: str | None,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(
        experiment_name, artifact_location=artifact_location
    )


def _flatten_config(
    config: dict[str, Any], prefix: str = ""
) -> Iterator[tuple[str, str]]:
    for name, value in config.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from _flatten_config(value, key)
        elif isinstance(value, str):
            yield key, value
        elif value is None:
            yield key, "null"
        elif isinstance(value, bool):
            yield key, str(value).lower()
        elif isinstance(value, (int, float)):
            yield key, str(value)
        else:
            yield key, yaml.safe_dump(value, default_flow_style=True).strip()


def log_checkpoint(
    client: Any,
    experiment_id: str,
    checkpoint_path: Path,
    model_name: str,
    unique_id: int,
    score: float,
    configs: dict[str, Any],
) -> str:
    run = client.create_run(
        experiment_id,
        run_name=f"{model_name}_{unique_id:06d}",
        tags={
            "serializer.model": model_name,
            "serializer.score": str(score),
        },
    )
    run_id = run.info.run_id
    try:
        client.log_artifacts(run_id, str(checkpoint_path))
        for name, value in _flatten_config(configs):
            client.log_param(run_id, name, value)
        with (checkpoint_path / "metrics.csv").open(
            newline="", encoding="utf-8"
        ) as metrics_file:
            for step, values in enumerate(csv.DictReader(metrics_file)):
                for name, value in values.items():
                    client.log_metric(run_id, name, float(value), step=step)
        client.set_terminated(run_id)
    except BaseException:
        client.delete_run(run_id)
        raise
    return run_id
