import subprocess
import sys

from model_serializer.tracking.ui import main


def test_ui_entrypoint_forwards_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model-serializer-ui",
            "--backend-store-uri",
            "sqlite:///models.db",
            "--port",
            "5001",
        ],
    )
    commands = []

    def run(command, check):
        commands.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)

    assert main() == 0
    assert commands == [
        (
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                "sqlite:///models.db",
                "--port",
                "5001",
            ],
            False,
        )
    ]
