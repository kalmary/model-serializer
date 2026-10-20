import math
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from numbers import Real


@dataclass
class Metrics(ABC):
    def __post_init__(self) -> None:
        if type(self) is Metrics:
            raise TypeError("Metrics must be subclassed")


def validate_metrics[MetricsType: Metrics](
    metrics: MetricsType,
    observed_metric: Callable[[MetricsType], list[float]],
) -> tuple[list[tuple[str, list[float]]], float]:
    if not isinstance(metrics, Metrics) or not is_dataclass(metrics):
        raise TypeError("metrics must be a Metrics dataclass")

    metric_fields = fields(metrics)
    if not metric_fields:
        raise ValueError("metrics must define at least one field")

    metric_values = []
    history_length = None
    for metric_field in metric_fields:
        history = getattr(metrics, metric_field.name)
        if not isinstance(history, list):
            raise TypeError("every metric field must be a list")
        if not history:
            raise ValueError("metric histories must be non-empty")
        if history_length is None:
            history_length = len(history)
        elif len(history) != history_length:
            raise ValueError("metric histories must have equal lengths")
        for value in history:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("metric histories must contain real numbers")
            if not math.isfinite(float(value)):
                raise ValueError("metric histories must contain finite values")
        metric_values.append((metric_field.name, history))

    observed_history = observed_metric(metrics)
    if not isinstance(observed_history, list):
        raise TypeError("observed metric must return a list")
    if not any(observed_history is history for _, history in metric_values):
        raise ValueError("observed metric must return a metrics field")
    if len(observed_history) != history_length:
        raise ValueError("observed metric has an invalid history length")
    return metric_values, float(observed_history[-1])
