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

Add the package from a local checkout to another uv project:

```bash
# Local serializer only
uv add --editable /path/to/model_serializer

# Local and MLflow serializers
uv add --editable --extra mlflow /path/to/model_serializer
```

## Metrics and model factory

Define the metrics collected by the training loop:

```python
from dataclasses import dataclass

from torch import nn

from model_serializer import Metrics


@dataclass
class TrainingMetrics(Metrics):
    train_loss: list[float]
    validation_loss: list[float]
    miou: list[float]


def build_model(configs: dict) -> nn.Module:
    return nn.Linear(**configs["model"])


configs = {
    "model": {
        "in_features": 4,
        "out_features": 1,
    }
}
```

The model factory is only needed when loading a checkpoint. It receives the same configuration mapping that was given to the serializer.

### Metric definition

The package has no predefined metrics. The application defines them as fields of a dataclass that subclasses `Metrics`. Each field name is used unchanged as a column in `metrics.csv` and, with MLflow tracking, as the MLflow metric name.

The serializer validates the shape and storage format of these histories:

- the metrics object must be a dataclass subclass of `Metrics`,
- it must declare at least one metric field,
- every field must be a non-empty list of finite real numbers,
- all histories must have the same length,
- `observed_metric` must return one of those field lists directly.

It does not enforce semantic rules such as a fixed set of names or a `0–1` range for IoU. Those rules belong to the training code that calculates each metric. The latest value in the selected history determines whether a checkpoint enters the top N; all histories are still saved.

## Local serializer

Create a serializer, append one value to every metric history after each epoch, and submit the current model:

```python
from model_serializer import ModelSerializer, load_checkpoint_model


model = build_model(configs)
metrics = TrainingMetrics([], [], [])

serializer = ModelSerializer[TrainingMetrics](
    mode="max",
    observed_metric=lambda metrics: metrics.miou,
    n_models=3,
    path="checkpoints",
    model_name="unet",
    configs=configs,
)

for epoch in range(number_of_epochs):
    train_loss, validation_loss, miou = train_one_epoch(model, epoch)
    metrics.train_loss.append(train_loss)
    metrics.validation_loss.append(validation_loss)
    metrics.miou.append(miou)
    serializer.checkpoint(model, metrics)

best_checkpoint = sorted(serializer.path.iterdir())[0]
loaded_model = load_checkpoint_model(
    best_checkpoint,
    model_factory=build_model,
)
```

`mode="max"` keeps the largest observed values; use `mode="min"` for metrics such as loss. `checkpoint()` returns the saved directory when the model enters the top N and `None` when it does not. Ranked directories can be renamed as better models arrive, so resolve the final best path after training as shown above.

## MLflow tracking

The MLflow serializer applies the same top-N selection while logging accepted checkpoints as MLflow runs. Use explicit paths so the database and artifacts do not depend on the process working directory:

```python
from pathlib import Path

from model_serializer.tracking import (
    MLflowModelSerializer,
    load_mlflow_model,
)


storage = Path("data/mlflow").resolve()
storage.mkdir(parents=True, exist_ok=True)
tracking_uri = f"sqlite:///{storage / 'mlflow.db'}"
artifact_location = (storage / "artifacts").as_uri()

model = build_model(configs)
metrics = TrainingMetrics([], [], [])
last_accepted_run_id = None

with MLflowModelSerializer(
    mode="max",
    observed_metric=lambda metrics: metrics.miou,
    n_models=3,
    model_name="unet",
    configs=configs,
    experiment_name="field-segm",
    tracking_uri=tracking_uri,
    artifact_location=artifact_location,
) as serializer:
    for epoch in range(number_of_epochs):
        train_loss, validation_loss, miou = train_one_epoch(model, epoch)
        metrics.train_loss.append(train_loss)
        metrics.validation_loss.append(validation_loss)
        metrics.miou.append(miou)

        run_id = serializer.checkpoint(model, metrics)
        if run_id is not None:
            last_accepted_run_id = run_id

assert last_accepted_run_id is not None
loaded_model = load_mlflow_model(
    last_accepted_run_id,
    model_factory=build_model,
    tracking_uri=tracking_uri,
)
```

`checkpoint()` returns an MLflow run ID when the model enters the top N and `None` otherwise. Configuration values are logged as parameters, metric histories are logged by their dataclass field names and list positions, and rank is stored in the `serializer.rank` tag. Runs displaced from the top N are soft-deleted. MLflow does not add or require a predefined set of training metrics.

Inspect the experiment in the MLflow UI:

```bash
uv run model-serializer-ui \
  --backend-store-uri sqlite:///data/mlflow/mlflow.db
```

Then open `http://127.0.0.1:5000`. `model-serializer-ui` is a thin entrypoint for MLflow's existing UI, so it accepts the same options, including `--host` and `--port`. The UI shows the serializer's runs, parameters, metric histories, ranks, and checkpoint artifacts. These checkpoints are MLflow runs; the serializer does not add them to the separate MLflow Model Registry.

The package stores inference checkpoints. Resumable training additionally requires optimizer, scheduler, epoch, gradient-scaler, and random-generator state.
