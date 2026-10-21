"""Fail if git tracks a file that could hold secrets (*.key, .env, .env.*, *.db, *.pem)."""
import fnmatch
import subprocess
import sys

FORBIDDEN = ("*.key", ".env", ".env.*", "*.db", "*.pem")
ALLOWED = (".env.example",)


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def find_forbidden(paths: list[str]) -> list[str]:
    bad = []
    for path in paths:
        name = _basename(path)
        if name in ALLOWED:
            continue
        if any(fnmatch.fnmatchcase(name, pat) for pat in FORBIDDEN):
            bad.append(path)
    return bad


def main() -> int:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    bad = find_forbidden([p for p in out.splitlines() if p])
    if bad:
        print("Forbidden files are tracked by git:")
        for path in bad:
            print(f"  {path}")
        return 1
    print("check_no_secrets: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
