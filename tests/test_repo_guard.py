import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_no_secrets import find_forbidden  # noqa: E402


def test_forbidden_patterns_detected():
    paths = ["a/.env", "x.key", "ok.py", "d/trader.db"]
    assert find_forbidden(paths) == ["a/.env", "x.key", "d/trader.db"]


def test_clean_list_passes():
    assert find_forbidden(["judge/data.py"]) == []


def test_windows_separators_and_other_patterns():
    paths = ["a\\.env.local", "certs\\site.pem", "k\\ssh-key.key", "a\\.env.example", ".env.example"]
    assert find_forbidden(paths) == ["a\\.env.local", "certs\\site.pem", "k\\ssh-key.key"]


def test_results_dir_is_not_gitignored():
    # results/ is the record and must be committed.
    import subprocess

    root = Path(__file__).resolve().parent.parent
    res = subprocess.run(["git", "check-ignore", "results/x-test.json", "results/x.final_opened"],
                         cwd=root, capture_output=True)
    assert res.returncode == 1, f"results/ files are gitignored: {res.stdout!r} {res.stderr!r}"
