import csv
import importlib
from dataclasses import dataclass

import pytest
import torch
import yaml
from torch import nn


def _serializer_module():
    return importlib.import_module("model_serializer")


def _metrics_type(module):
    @dataclass
    class TrainingMetrics(module.Metrics):
        accuracy: list[float]
        miou: list[float]

    return TrainingMetrics


def _model(value):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def test_checkpoint_keeps_and_reorders_best_max_models(tmp_path):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    output_path = tmp_path / "checkpoints"
    configs = {"train": {"epochs": 3}, "input": {"channels": ["red", "nir"]}}
    serializer = module.ModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: metrics.miou,
        n_models=2,
        path=output_path,
        model_name="unet",
        configs=configs,
    )

    serializer.checkpoint(_model(1.0), TrainingMetrics([0.4], [0.6]))
    serializer.checkpoint(_model(2.0), TrainingMetrics([0.3], [0.4]))
    saved_path = serializer.checkpoint(
        _model(3.0),
        TrainingMetrics([0.5, 0.7], [0.6, 0.8]),
    )

    assert saved_path == output_path / "01_unet_000003"
    assert sorted(path.name for path in output_path.iterdir()) == [
        "01_unet_000003",
        "02_unet_000001",
    ]
    state_dict = torch.load(saved_path / "model.pt", weights_only=True)
    torch.testing.assert_close(state_dict["weight"], torch.tensor([[3.0]]))
    with (saved_path / "metrics.csv").open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file)) == [
            {"accuracy": "0.5", "miou": "0.6"},
            {"accuracy": "0.7", "miou": "0.8"},
        ]
    with (saved_path / "configs.yaml").open(encoding="utf-8") as file:
        assert yaml.safe_load(file) == configs


def test_checkpoint_rebuilds_model_with_factory(tmp_path):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    configs = {
        "model": {"in_features": 1, "out_features": 1, "bias": False},
    }
    model = _model(3)
    serializer = module.ModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: metrics.miou,
        n_models=1,
        path=tmp_path / "checkpoints",
        model_name="unet",
        configs=configs,
    )
    saved_path = serializer.checkpoint(
        model, TrainingMetrics([0.5], [0.5])
    )

    loaded = module.load_checkpoint_model(
        saved_path, lambda saved: nn.Linear(**saved["model"])
    )

    assert type(loaded) is type(model)
    for expected, actual in zip(
        model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_rejected_model_still_advances_unique_id(tmp_path):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    serializer = module.ModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: metrics.miou,
        n_models=1,
        path=tmp_path / "checkpoints",
        model_name="unet",
        configs={},
    )

    serializer.checkpoint(_model(1.0), TrainingMetrics([0.6], [0.6]))
    rejected = serializer.checkpoint(_model(2.0), TrainingMetrics([0.5], [0.5]))
    saved = serializer.checkpoint(_model(3.0), TrainingMetrics([0.7], [0.7]))

    assert rejected is None
    assert saved.name == "01_unet_000003"


def test_checkpoint_supports_min_mode_and_keeps_earlier_tie(tmp_path):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    serializer = module.ModelSerializer[TrainingMetrics](
        mode="min",
        observed_metric=lambda metrics: metrics.miou,
        n_models=1,
        path=tmp_path / "checkpoints",
        model_name="unet",
        configs={},
    )

    serializer.checkpoint(_model(1.0), TrainingMetrics([0.3], [0.3]))
    tied = serializer.checkpoint(_model(2.0), TrainingMetrics([0.3], [0.3]))
    saved = serializer.checkpoint(_model(3.0), TrainingMetrics([0.2], [0.2]))

    assert tied is None
    assert saved.name == "01_unet_000003"


def test_constructor_clears_output_directory(tmp_path):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    output_path = tmp_path / "checkpoints"
    output_path.mkdir()
    (output_path / "old-file.txt").write_text("old", encoding="utf-8")

    module.ModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: metrics.miou,
        n_models=1,
        path=output_path,
        model_name="unet",
        configs={},
    )

    assert list(output_path.iterdir()) == []


@pytest.mark.parametrize(
    "metrics_factory",
    [
        lambda metrics_type: metrics_type([], []),
        lambda metrics_type: metrics_type([0.5], [0.5, 0.6]),
        lambda metrics_type: metrics_type([0.5], [float("nan")]),
    ],
)
def test_checkpoint_rejects_invalid_metric_histories(tmp_path, metrics_factory):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    serializer = module.ModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: metrics.miou,
        n_models=1,
        path=tmp_path / "checkpoints",
        model_name="unet",
        configs={},
    )

    with pytest.raises(ValueError):
        serializer.checkpoint(_model(1.0), metrics_factory(TrainingMetrics))


def test_checkpoint_rejects_selector_result_not_owned_by_metrics(tmp_path):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    serializer = module.ModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: list(metrics.miou),
        n_models=1,
        path=tmp_path / "checkpoints",
        model_name="unet",
        configs={},
    )

    with pytest.raises(ValueError):
        serializer.checkpoint(_model(1.0), TrainingMetrics([0.5], [0.5]))


def test_failed_serialization_preserves_ranking_and_consumes_id(tmp_path):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    output_path = tmp_path / "checkpoints"
    serializer = module.ModelSerializer[TrainingMetrics](
        mode="max",
        observed_metric=lambda metrics: metrics.miou,
        n_models=1,
        path=output_path,
        model_name="unet",
        configs={},
    )
    serializer.checkpoint(_model(1.0), TrainingMetrics([0.5], [0.5]))

    class InvalidModel:
        def state_dict(self):
            raise RuntimeError("cannot serialize")

    with pytest.raises(RuntimeError):
        serializer.checkpoint(InvalidModel(), TrainingMetrics([0.6], [0.6]))

    assert [path.name for path in output_path.iterdir()] == ["01_unet_000001"]
    saved = serializer.checkpoint(_model(3.0), TrainingMetrics([0.7], [0.7]))
    assert saved.name == "01_unet_000003"


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("mode", "largest", ValueError),
        ("n_models", 0, ValueError),
        ("model_name", "models/unet", ValueError),
    ],
)
def test_constructor_rejects_invalid_arguments(tmp_path, argument, value, error):
    module = _serializer_module()
    TrainingMetrics = _metrics_type(module)
    arguments = {
        "mode": "max",
        "observed_metric": lambda metrics: metrics.miou,
        "n_models": 1,
        "path": tmp_path / "checkpoints",
        "model_name": "unet",
        "configs": {},
    }
    arguments[argument] = value

    with pytest.raises(error):
        module.ModelSerializer[TrainingMetrics](**arguments)
