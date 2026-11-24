"""The regression check to run before every change is kept (CI runs it on each push): the fast
test suite (seconds, no AI models).

    python scripts/check.py

Exit code 0 = safe to keep the change.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, args: list[str]) -> bool:
    print(f"\n== {label}", flush=True)
    return subprocess.run([sys.executable, *args], cwd=ROOT).returncode == 0


if __name__ == "__main__":
    ok = run("fast tests", ["-m", "pytest", "-m", "not slow", "-q", "-p", "no:cacheprovider"])
    print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED: don't keep this change as it is")
    sys.exit(0 if ok else 1)
