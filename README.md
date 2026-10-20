# model-serializer

`model-serializer` keeps the best PyTorch checkpoints according to a selected metric. Every checkpoint stores model weights, complete metric histories, and its resolved configuration together.

```text
01_model_000003/
├── model.pt
├── metrics.csv
└── configs.yaml
```

## Installation

Install the local package and its test dependencies:

```bash
uv sync --extra mlflow --group test
```

The `mlflow` extra is only required for MLflow tracking.

## Local checkpoints

Define the metrics collected by the training loop:

```python
from dataclasses import dataclass

from model_serializer import Metrics, ModelSerializer


@dataclass
class TrainingMetrics(Metrics):
    loss: list[float]
    miou: list[float]
```

Create a serializer and submit the model after an epoch:

```python
serializer = ModelSerializer[TrainingMetrics](
    mode="max",
    observed_metric=lambda metrics: metrics.miou,
    n_models=3,
    path="checkpoints",
    model_name="unet",
    configs=configs,
)

serializer.checkpoint(model, metrics)
```

The package does not prescribe metric names. Every metric field must contain a numeric history of the same length. The selected field's latest value determines checkpoint ranking.

Loading is model-agnostic. Supply a factory that rebuilds a model from the saved configuration:

```python
from model_serializer import load_checkpoint_model

model = load_checkpoint_model(
    "checkpoints/01_unet_000003",
    model_factory=lambda configs: build_model(configs),
)
```

## MLflow tracking

The MLflow serializer applies the same top-N selection while logging accepted checkpoints as MLflow runs:

```python
from model_serializer.tracking import MLflowModelSerializer

with MLflowModelSerializer(
    mode="max",
    observed_metric=lambda metrics: metrics.miou,
    n_models=3,
    model_name="unet",
    configs=configs,
    experiment_name="segmentation",
    tracking_uri="sqlite:///mlflow.db",
    artifact_location="file:./mlartifacts",
) as serializer:
    serializer.checkpoint(model, metrics)
```

Configuration values are logged as parameters, metric histories are logged by step, and rank is stored in the `serializer.rank` tag. Runs displaced from the top N are soft-deleted.

The package stores inference checkpoints. Resumable training additionally requires optimizer, scheduler, epoch, gradient-scaler, and random-generator state.
