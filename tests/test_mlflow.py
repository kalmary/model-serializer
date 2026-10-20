from dataclasses import dataclass
from pathlib import Path

import torch
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from torch import nn

from model_serializer import Metrics
from model_serializer.tracking import (
    MLflowModelSerializer,
    load_mlflow_model,
)


@dataclass
class TrainingMetrics(Metrics):
    loss: list[float]
    miou: list[float]


def model(value: float) -> torch.nn.Module:
    result = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        next(result.parameters()).fill_(value)
    return result


def test_mlflow_serializer_logs_and_loads_ranked_checkpoints(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = (tmp_path / "artifacts").as_uri()
    configs = {
        "model": {"in_features": 1, "out_features": 1, "bias": False},
    }
    with MLflowModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: metrics.miou,
        n_models=2,
        model_name="unet",
        configs=configs,
        experiment_name="field-segm-test",
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
    ) as serializer:
        first = serializer.checkpoint(
            model(1), TrainingMetrics([0.8], [0.5])
        )
        removed = serializer.checkpoint(
            model(2), TrainingMetrics([0.9], [0.4])
        )
        best = serializer.checkpoint(
            model(3), TrainingMetrics([0.8, 0.6], [0.5, 0.7])
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("field-segm-test")
    runs = client.search_runs(
        [experiment.experiment_id], run_view_type=ViewType.ALL
    )
    stages = {run.info.run_id: run.info.lifecycle_stage for run in runs}

    assert stages[first] == "active"
    assert stages[best] == "active"
    assert stages[removed] == "deleted"
    assert {artifact.path for artifact in client.list_artifacts(best)} == {
        "configs.yaml",
        "metrics.csv",
        "model.pt",
    }
    assert [
        metric.value for metric in client.get_metric_history(best, "miou")
    ] == [0.5, 0.7]
    assert client.get_run(best).data.tags["serializer.rank"] == "1"
    assert client.get_run(best).data.params["model.in_features"] == "1"

    loaded = load_mlflow_model(
        best,
        lambda saved: nn.Linear(**saved["model"]),
        tracking_uri=tracking_uri,
    )
    torch.testing.assert_close(
        next(loaded.parameters()), next(model(3).parameters())
    )
