import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_attribution", Path(__file__).resolve().parent.parent / "scripts" / "check_attribution.py"
)
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)

TOOL = "Cla" + "ude"
TRAILER = "Co-" + "Authored-By: " + TOOL + " Opus <noreply@" + "anthro" + "pic.com>"


def test_clean_message_passes():
    assert ca.bad_text("feat(judge): trade diagnostics in every result file") == []


def test_coauthor_trailer_blocked():
    assert ca.bad_text("fix: thing\n\n" + TRAILER)


def test_any_coauthor_trailer_blocked():
    assert ca.bad_text("fix: thing\n\nCo-" + "authored-by: Someone <a@b.c>")


def test_generated_with_line_blocked():
    assert ca.bad_text("Summary\n\nGenerated " + "with " + TOOL + " Code")


def test_tool_name_in_message_blocked():
    assert ca.bad_text("docs: notes from " + TOOL.lower())


def test_word_inside_other_word_not_blocked():
    assert ca.bad_text("docs: mention Claudette's rule") == []


def test_identity_blocked():
    assert ca.bad_identity(TOOL + " <noreply@" + "anthro" + "pic.com>")
    assert ca.bad_identity("Haikeys <haikeysfpl@gmail.com>") == []


def test_check_commits_reports_each_problem():
    rows = [
        ("a" * 40, "Haikeys <h@x.com>", "Haikeys <h@x.com>", "feat: ok"),
        ("b" * 40, "Haikeys <h@x.com>", "GitHub <noreply@github.com>", "fix: x\n\n" + TRAILER),
        ("c" * 40, TOOL + " <c@x.com>", "Haikeys <h@x.com>", "feat: y"),
    ]
    problems = ca.check_commits(rows)
    assert any(p.startswith("bbbbbbb message") for p in problems)
    assert any(p.startswith("ccccccc author") for p in problems)
    assert not any(p.startswith("aaaaaaa") for p in problems)


def test_check_files_blocks_trailer_and_skips_guard_files():
    files = {
        "README.md": "Built by Haikeys.\n\nGenerated " + "with " + TOOL + " Code",
        "notes.md": "plain text",
        "scripts/check_attribution.py": TRAILER,
    }
    problems = ca.check_files(list(files), files.__getitem__)
    assert problems == [p for p in problems if p.startswith("README.md")] and len(problems) == 1


def test_repo_history_and_files_are_clean():
    assert ca.check_commits(ca._commits()) == []


def test_message_file_mode(tmp_path, monkeypatch):
    msg = tmp_path / "MSG"
    msg.write_text("fix: thing\n\n" + TRAILER + "\n# comment lines are ignored\n", encoding="utf-8")
    assert ca.main(["--message-file", str(msg)]) == 1
    msg.write_text("fix: thing\n# Co-" + "Authored-By in a comment is ignored\n", encoding="utf-8")
    assert ca.main(["--message-file", str(msg)]) == 0


def test_pr_body_mode(monkeypatch):
    monkeypatch.setenv("PR_BODY", "## Checks\n\nGenerated " + "with " + TOOL + " Code")
    assert ca.main(["--pr-body-env", "PR_BODY"]) == 1
    monkeypatch.setenv("PR_BODY", "## Checks\n\nAll tests pass.")
    assert ca.main(["--pr-body-env", "PR_BODY"]) == 0
