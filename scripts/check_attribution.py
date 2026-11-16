"""Fail if any commit, commit message, PR description or tracked file credits an AI tool.

The repo owner is the only author and contributor. This blocks co-author
trailers, tool names in commit messages or identities, and "generated with"
lines. Patterns are assembled from pieces so this file never matches itself.

Usage:
  python scripts/check_attribution.py                  # every commit reachable from HEAD + tracked files
  python scripts/check_attribution.py --message-file F # one commit message (commit-msg hook)
  PR_BODY=... python scripts/check_attribution.py --pr-body-env PR_BODY
"""
import argparse
import os
import re
import subprocess
import sys

_TOOL = "cla" + "ude"
_VENDOR = "anthro" + "pic"
_COAUTHOR = "co-" + "authored-by"
_GENERATED = "generated" + r"\s+with\s+" + _TOOL

MESSAGE_PATTERNS = [
    re.compile(_COAUTHOR, re.IGNORECASE),
    re.compile(rf"\b{_TOOL}\b", re.IGNORECASE),
    re.compile(_VENDOR, re.IGNORECASE),
    re.compile(_GENERATED, re.IGNORECASE),
]
IDENTITY_PATTERNS = [
    re.compile(_TOOL, re.IGNORECASE),
    re.compile(_VENDOR, re.IGNORECASE),
]
FILE_PATTERNS = [
    re.compile(_COAUTHOR + r":\s*" + _TOOL, re.IGNORECASE),
    re.compile(_GENERATED, re.IGNORECASE),
]
# The guard and its tests necessarily describe what they block.
FILE_EXEMPT = {"scripts/check_attribution.py", "tests/test_attribution.py", ".githooks/commit-msg"}


def bad_text(text: str, patterns=MESSAGE_PATTERNS) -> list[str]:
    return [m.group(0) for p in patterns for m in [p.search(text)] if m]


def bad_identity(ident: str) -> list[str]:
    return bad_text(ident, IDENTITY_PATTERNS)


def _commits(rev: str = "HEAD") -> list[tuple[str, str, str, str]]:
    out = subprocess.run(
        ["git", "log", rev, "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x1e"],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout
    rows = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if rec:
            sha, author, committer, body = rec.split("\x00", 3)
            rows.append((sha, author, committer, body))
    return rows


def check_commits(rows) -> list[str]:
    problems = []
    for sha, author, committer, body in rows:
        for hit in bad_identity(author):
            problems.append(f"{sha[:7]} author '{author}' matches '{hit}'")
        for hit in bad_identity(committer):
            problems.append(f"{sha[:7]} committer '{committer}' matches '{hit}'")
        for hit in bad_text(body):
            problems.append(f"{sha[:7]} message contains '{hit}'")
    return problems


def check_files(paths: list[str], read) -> list[str]:
    problems = []
    for path in paths:
        if path.replace("\\", "/") in FILE_EXEMPT:
            continue
        try:
            text = read(path)
        except (OSError, UnicodeDecodeError):
            continue
        for hit in bad_text(text, FILE_PATTERNS):
            problems.append(f"{path} contains '{hit}'")
    return problems


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message-file")
    ap.add_argument("--pr-body-env")
    args = ap.parse_args(argv)

    problems: list[str] = []
    if args.message_file:
        with open(args.message_file, encoding="utf-8") as fh:
            msg = "\n".join(l for l in fh.read().splitlines() if not l.startswith("#"))
        problems += [f"commit message contains '{h}'" for h in bad_text(msg)]
        for var in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
            ident = subprocess.run(["git", "var", var], capture_output=True, text=True).stdout
            problems += [f"{var} '{ident.strip()}' matches '{h}'" for h in bad_identity(ident)]
    elif args.pr_body_env:
        problems += [f"PR description contains '{h}'" for h in bad_text(os.environ.get(args.pr_body_env, ""))]
    else:
        problems += check_commits(_commits())
        tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
        problems += check_files([p for p in tracked.splitlines() if p], _read)

    if problems:
        print("check_attribution: FAILED. The repo owner is the only author; remove these:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("check_attribution: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
