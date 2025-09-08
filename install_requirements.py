import subprocess
import sys
from pathlib import Path

def main() -> None:
    repo_root = Path(__file__).resolve().parent
    requirements = repo_root / "requirements.txt"
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements),
    ])

if __name__ == "__main__":
    main()