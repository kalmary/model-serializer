import subprocess
import sys


def main() -> int:
    command = [sys.executable, "-m", "mlflow", "ui", *sys.argv[1:]]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
