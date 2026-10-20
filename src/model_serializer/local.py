import shutil
import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from model_serializer.artifacts import (
    load_checkpoint_model,
    validate_model,
    write_checkpoint_artifacts,
)
from model_serializer.metrics import Metrics, validate_metrics
from model_serializer.ranking import (
    Checkpoint,
    assign_ranked_paths,
    qualifies,
    rank_checkpoints,
)
from model_serializer.validation import (
    validate_output_path,
    validate_serializer,
)


class ModelSerializer[MetricsType: Metrics]:
    def __init__(
        self,
        mode: Literal["min", "max"],
        observed_metric: Callable[[MetricsType], list[float]],
        n_models: int,
        path: str | Path,
        model_name: str,
        configs: dict[str, Any],
    ) -> None:
        validate_serializer(mode, observed_metric, n_models, model_name, configs)
        self.mode = mode
        self.observed_metric = observed_metric
        self.n_models = n_models
        self.model_name = model_name
        self.configs = deepcopy(configs)
        self.path = Path(path).expanduser().resolve()
        validate_output_path(self.path)
        self._rank_width = max(2, len(str(n_models)))
        self._checked_models = 0
        self._checkpoints: list[Checkpoint] = []

        if self.path.exists():
            if not self.path.is_dir():
                raise NotADirectoryError(
                    f"checkpoint path is not a directory: {self.path}"
                )
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)

    def checkpoint(
        self,
        model: Any,
        metrics: MetricsType,
        configs: dict[str, Any] | None = None,
    ) -> Path | None:
        metric_values, score = validate_metrics(metrics, self.observed_metric)
        validate_model(model)
        if configs is not None and not isinstance(configs, dict):
            raise TypeError("configs must be a dictionary")
        checkpoint_configs = self.configs if configs is None else deepcopy(configs)
        self._checked_models += 1
        unique_id = self._checked_models
        if not qualifies(self._checkpoints, score, self.mode, self.n_models):
            return None

        temporary_path = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=self.path))
        try:
            write_checkpoint_artifacts(
                temporary_path, model, metric_values, checkpoint_configs
            )
        except BaseException:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        candidate = Checkpoint(score, unique_id, temporary_path)
        ranked, removed = rank_checkpoints(
            self._checkpoints,
            candidate,
            self.mode,
            self.n_models,
        )
        if removed is not None:
            shutil.rmtree(removed.path)

        assign_ranked_paths(ranked, self.path, self.model_name, self._rank_width)
        self._checkpoints = ranked
        return next(item.path for item in ranked if item.unique_id == unique_id)


__all__ = ["Metrics", "ModelSerializer", "load_checkpoint_model"]
