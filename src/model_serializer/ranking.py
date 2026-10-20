from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4


@dataclass
class Checkpoint:
    score: float
    unique_id: int
    path: Path


def qualifies(
    checkpoints: list[Checkpoint],
    score: float,
    mode: Literal["min", "max"],
    limit: int,
) -> bool:
    if len(checkpoints) < limit:
        return True
    worst_score = checkpoints[-1].score
    return score > worst_score if mode == "max" else score < worst_score


def rank_checkpoints(
    checkpoints: list[Checkpoint],
    candidate: Checkpoint,
    mode: Literal["min", "max"],
    limit: int,
) -> tuple[list[Checkpoint], Checkpoint | None]:
    ranked = [Checkpoint(item.score, item.unique_id, item.path) for item in checkpoints]
    ranked.append(candidate)
    ranked.sort(key=lambda item: item.score, reverse=mode == "max")
    removed = ranked.pop() if len(ranked) > limit else None
    return ranked, removed


def assign_ranked_paths(
    checkpoints: list[Checkpoint],
    root: Path,
    model_name: str,
    rank_width: int,
) -> None:
    staging_token = uuid4().hex
    for checkpoint in checkpoints:
        staging_path = root / f".ranking_{staging_token}_{checkpoint.unique_id}"
        checkpoint.path.rename(staging_path)
        checkpoint.path = staging_path

    for rank, checkpoint in enumerate(checkpoints, start=1):
        final_path = root / (
            f"{rank:0{rank_width}d}_{model_name}_{checkpoint.unique_id:06d}"
        )
        checkpoint.path.rename(final_path)
        checkpoint.path = final_path


def checkpoint_id(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[1])
