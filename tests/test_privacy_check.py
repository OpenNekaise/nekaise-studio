"""privacy_check — the rules that keep proprietary/personal strings out of history.

Fixture strings are split across adjacent literals ("/home/" "alice/...") so this file's
OWN source never contains a full flagged pattern — otherwise the pre-commit hook (rightly)
refuses to commit the test suite. Keep it that way when adding cases.
"""
from conftest import REPO, load_module

pc = load_module(REPO / "tools" / "privacy_check.py")
TERMS = ["securebuilding9"]


def hits(line):
    return pc.scan_line(line, TERMS)


def test_personal_paths_flagged():
    assert hits("source /home/" "alice/.env")
    assert hits("path = /Users/" "bob/data/x")
    assert not hits("relative/path/file.txt")


def test_tailscale_hosts_flagged_placeholders_exempt():
    assert hits("host = mybox.tail1234" ".ts.net")
    assert not hits("e.g. NEKAISE_DASH_HOSTS=.tailXXXX" ".ts.net")   # doc placeholder


def test_secret_shapes_flagged():
    assert hits("key = sk-ant-" "abcdefghijklmnop-qrstuvwx")
    assert hits("token ghp" "_" + "a" * 36)
    assert not hits("skill = run-experiment")                        # 'sk-' lookalikes stay quiet


def test_denylisted_terms_flagged_case_insensitively():
    assert hits("trained on SecureBuilding9 data")
    assert not hits("trained on some building data")


def test_privacy_ok_marker_exempts_line():
    assert not hits("documented exception /home/" "ci-runner/ # privacy-ok")
